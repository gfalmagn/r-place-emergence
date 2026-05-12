import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from rplacem import var as var
import rplacem.utilities as util
import rplacem.entropy as entropy
import rplacem.fractal_dim as fractal_dim
import warnings
from memory_profiler import profile
from scipy.spatial.distance import pdist
from scipy.signal import fftconvolve
import copy

def count_image_differences(pixels1, pixels2, npix, coor_idx, indices=None):
    ''' Count the number of pixels (at given *indices* of coordinates of cpart *cpart*) that differ
    between *pixels1* and *pixels2* (both 2d numpy arrays of shape (num y coords, num x coords)).
    coor_idx is cpart.coords_offset'''
    if indices is None:
        indices = np.arange(0, npix)
    coords = coor_idx[:, indices]
    return np.count_nonzero(pixels2[coords[1], coords[0]] - pixels1[coords[1], coords[0]])

def initialize_start_time_grid(cpart, t_start, add_color_dim=False):
    '''
    Initialize to the start of the time interval.
    We neglect the time before the supplementary canvas
    sections open by setting last_time_changed for those coordinates
    to either the interval start time or the time the coordinates
    become available, whichever happens later.
    '''
    if add_color_dim:
        res = np.full((var.NUM_COLORS, cpart.num_pix()), t_start, dtype='float64')
        for e in range(1, var.N_ENLARGE):
            res[:, cpart.canvassection_coordinds[e]] = max(t_start, var.TIME_ENLARGE[e])
    else:
        res = np.full(cpart.num_pix(), t_start, dtype='float64')
        for e in range(1, var.N_ENLARGE):
            res[cpart.canvassection_coordinds[e]] = max(t_start, var.TIME_ENLARGE[e])
    return res

#@profile
def calc_time_spent_in_color(cpart, t_inds,
                             t_beg, t_end, current_color,
                             last_time_installed_sw=None, last_time_removed_sw=None):
    '''
    Returns np.array of shape (n_pixels, n_colors)
    For each timestep, it contains the time that each pixel spent in each color
    Also updates last_time_installed_sw pointer.
    '''
    time_spent_in_color = np.zeros((cpart.num_pix(), var.NUM_COLORS), dtype=('float32' if (t_end - t_beg) < 1000 else 'float64'))

    # Initialize last_time_changed
    last_time_changed = initialize_start_time_grid(cpart, t_beg, add_color_dim=False)

    # Loop through each pixel change in the step, indexing by time.
    for s, c, coor_idx in zip(cpart.time(t_inds), cpart.color(t_inds), cpart.coordidx(t_inds)):
        # nothing necessary here if the color is not changed (redundant pixel change)
        if c == current_color[coor_idx]:
            continue

        # Add the time that this pixel spent in the most recent color.
        time_spent_in_color[coor_idx, current_color[coor_idx]] += s - last_time_changed[coor_idx]

        # Update the time of the last pixel change for this pixel.
        # This code updates the pointer last_time_installed_sw without returning.
        last_time_changed[coor_idx] = s
        if last_time_installed_sw is not None:
            last_time_installed_sw[c][coor_idx] = s # time where this pixel was changed to new color [c]
            last_time_removed_sw[current_color[coor_idx]][coor_idx] = s  # time where this pixel lost its old color [current_color]

        # Update the current color.
        current_color[coor_idx] = c

    # Add the time spent in the final color (from the last pixel change to the end-time).
    time_spent_in_color[np.arange(0, cpart.num_pix()), current_color] += np.maximum(t_end - last_time_changed, 0)

    del last_time_changed
    return time_spent_in_color


def cumulative_attack_timefrac(time_spent_in_col, ref_color, inds_active_pix, stepwidth):
    '''
    From the 2D array time_spent_in_col (shape (n_pixels, n_colors)) and the 1D image ref_color,
    computes the sum of the times that every pixel spent in another color than that of ref_color.
    Normalized by the total time summed over all pixels
    '''
    if len(inds_active_pix) == 0:
        res = 0
    else:
        time_spent = time_spent_in_col[inds_active_pix, :]
        time_spent_in_refcol = time_spent[ np.arange(0, time_spent.shape[0]), ref_color[inds_active_pix]]

        res = 1 - np.sum(time_spent_in_refcol) / (stepwidth * len(inds_active_pix))

    return max(res, 0)

def calc_stability(stable_timefrac, inds_active, compute_average):
    stab_per_pixel = stable_timefrac[:, 0]  # Get time fraction for most stable color.

    # Only use coordinates for which the interval intersects with the 'active' timerange for the composition.
    keepinds = np.intersect1d(np.where(stab_per_pixel > 1e-10), inds_active, assume_unique=True).astype(int)  # Remove indices with stab <= 0.
    stab_per_pixel = stab_per_pixel[keepinds]
    # Get time fraction for second most stable color, divided by time fraction of most stable color
    stab_per_pixel_runnerup = stable_timefrac[keepinds, 1] / stab_per_pixel
    # Get number of used colors for each pixel in this time interval
    num_used_colors_perpix = np.count_nonzero(stable_timefrac[keepinds, :], axis=1)

    # Now get stability averaged over active pixels.
    if compute_average:
        stability = util.pixel_integrated_stats(stab_per_pixel, 1, True)
        runnerup_timeratio = util.pixel_integrated_stats(stab_per_pixel_runnerup, 0)
        n_used_colors = util.pixel_integrated_stats(num_used_colors_perpix, 1)
    else:
        stability = stab_per_pixel
        runnerup_timeratio = stab_per_pixel_runnerup
        n_used_colors = num_used_colors_perpix

    return stability, runnerup_timeratio, n_used_colors

def label_edge_map(label_img):
    dy, dx = np.gradient(label_img.astype(float))
    edges = (dx != 0) | (dy != 0)
    return edges.astype(np.uint8)


def calc_stable_cols(time_spent_in_col):
    return np.flip(np.argsort(time_spent_in_col, axis=1), axis=1)


def substract_and_sparse_replace(time_spent_in_color_sw, time_spent_in_color_vst, time_spent_in_color, i_replace,
                                 pos_in_timespentvst, active_colors_vst, startidx_timespent, endidx_timespent,
                                 rec_len_regime, rec_lengths, rec_timespent_num, sw_width,
                                 replace=True, try_reduce=False):
    '''
    Substracts the [time_spent_in_color] from [time_spent_in_color_sw]
    Replaces [time_spent_in_color_vst] at position [i_replace] by [time_spent_in_color]
    Also keeps track of [active_colors_vst] at index [i_replace], [pos_in_timespentvst], [rec_len_regime], [rec_timespent_num], [startidx_timespent], [endidx_timespent]
    '''
    # boolean array of pixels active at the timestep that must be removed or added
    active_pix_removed = active_colors_vst[i_replace]
    num_cols_removed = np.count_nonzero(active_pix_removed, axis=1) # shape (n_pixels)
    if replace:
        active_pix_added = (time_spent_in_color >= 1e-3)
        num_cols_added = np.count_nonzero(active_pix_added, axis=1) # shape (n_pixels)

    # loop on the different 'regimes' of length of time_spent_in_color_vst
    for reg in [0, 1, 2]:
        pix_thisreg = (rec_len_regime == reg)
        if not np.any(pix_thisreg):
            continue

        # Here, substract old elements from time_spent_in_color_sw (time_spent summed in sliding window)
        for i_rmv in np.arange(np.max(num_cols_removed[pix_thisreg])): # max 32 iterations (1 per removed color)
            # indices of pixels that needs at least i_rmv removed, and have the right length of recorded timespent (reg)
            enough_removedcols = np.where((num_cols_removed >= i_rmv + 1) & pix_thisreg)[0]
            # substract from time_spent_in_color_sw
            time_spent_in_color_sw[enough_removedcols,
                                   nth_true_element(active_pix_removed[enough_removedcols], i_rmv)] -= time_spent_in_color_vst[reg][pos_in_timespentvst[enough_removedcols],
                                                                                                        (startidx_timespent[enough_removedcols] + i_rmv) % rec_lengths[reg]]

        if not replace:
            continue
        # Here, add new elements in time_spent_in_color_vst, corresponding to the active colors in the new timestep
        for i_add in np.arange(np.max(num_cols_added[pix_thisreg])): # max 32 iterations (1 per removed color)
            # indices of pixels that needs at least i_add added, and have the right length of recorded timespent (reg)
            enough_addedcols = np.where((num_cols_added >= i_add + 1) & (rec_len_regime == reg))[0]
            # add to time_spent_in_color_vst
            time_spent_in_color_vst[reg][pos_in_timespentvst[enough_addedcols],
                                        (endidx_timespent[enough_addedcols] + i_add) % rec_lengths[reg]] = time_spent_in_color[enough_addedcols,
                                                                                                                               nth_true_element(active_pix_added[enough_addedcols], i_add)]

    # deplace pointer for the start and end of useful values in time_spent_in_color_vst
    startidx_timespent += num_cols_removed
    startidx_timespent = startidx_timespent % rec_lengths[rec_len_regime]
    if not replace:
        return
    endidx_timespent += num_cols_added
    endidx_timespent = endidx_timespent % rec_lengths[rec_len_regime]

    # number of useful values, indicator of need to use larger array
    rec_timespent_num += num_cols_added - num_cols_removed

    # renew the boolean array active_colors
    active_colors_vst[i_replace] = np.copy(active_pix_added)

    # add length to rec_timesepent_len when endidx > startidx + rec_length
    # the condition is based on the number of added colors in the next step (i_replace + 1)
    num_cols_tocome = np.count_nonzero(active_colors_vst[(i_replace+1) % sw_width], axis=1)
    for j in np.where(rec_timespent_num > rec_lengths[rec_len_regime] - num_cols_tocome)[0]:
        time_spent_in_color_vst[rec_len_regime[j]+1] = np.append(time_spent_in_color_vst[rec_len_regime[j]+1],
                                                                [np.concatenate(( time_spent_in_color_vst[rec_len_regime[j]][pos_in_timespentvst[j]],
                                                                                np.zeros((rec_lengths[rec_len_regime[j]+1]-rec_lengths[rec_len_regime[j]])) ),
                                                                                axis=0 )],
                                                                axis=0 )

        # remove the array from the rec_len_regime=1 arrays
        if rec_len_regime[j] > 0:
            np.delete(time_spent_in_color_vst[rec_len_regime[j]], pos_in_timespentvst[j], axis=0)
            pos_in_timespentvst[np.where((rec_len_regime == j) & (pos_in_timespentvst > pos_in_timespentvst[j]))] -= 1

        # change position of this pixel in the time_spent_in_color_vst[rec_len_regime[j]+1] array. This position is stored in pos_in_timespentvst
        pos_in_timespentvst[j] = time_spent_in_color_vst[rec_len_regime[j]+1].shape[0] - 1 # contains an array [1,2,...,n_pixels] where(reg==0), else contains position for this pixel in time_spent_vst[reg>0]
        rec_len_regime[j] += 1

    # try reducing the arrays to smaller sizes
    if try_reduce:
        print('Before reducing, There are',time_spent_in_color_vst[1].shape[0],'pixels in reg',1)
        print('remove',np.count_nonzero((rec_len_regime == 1) & (rec_timespent_num < (rec_lengths[0] - num_cols_tocome))),'pixels from regime',1)
        print('remove',np.count_nonzero((rec_len_regime == 2) & (rec_timespent_num < (rec_lengths[1] - num_cols_tocome))),'pixels from regime',2)
        for j in np.where((rec_len_regime > 0) & (rec_timespent_num < (rec_lengths[np.maximum(0, rec_len_regime - 1)] - num_cols_tocome)))[0]:
            new_reg = rec_len_regime[j] - 1
            if rec_len_regime[j] == 1:
                time_spent_in_color_vst[0][j] = time_spent_in_color_vst[1][pos_in_timespentvst[j]][0:rec_lengths[0]]
            else:
                time_spent_in_color_vst[new_reg] = np.append(time_spent_in_color_vst[new_reg],
                                                                          [time_spent_in_color_vst[rec_len_regime[j]][pos_in_timespentvst[j]][0:rec_lengths[new_reg]]], axis=0 )

            time_spent_in_color_vst[rec_len_regime[j]] = np.delete(time_spent_in_color_vst[rec_len_regime[j]], pos_in_timespentvst[j], axis=0)
            pos_in_timespentvst[np.where((rec_len_regime == rec_len_regime[j]) & (pos_in_timespentvst > pos_in_timespentvst[j]))] -= 1

            pos_in_timespentvst[j] = j if (new_reg == 0) else (time_spent_in_color_vst[new_reg].shape[0] - 1) # contains an array [1,2,...,n_pixels] where(reg==0), else contains position for this pixel in time_spent_vst[reg>0]
            rec_len_regime[j] -= 1

    del active_pix_added, active_pix_removed, num_cols_added, num_cols_removed
    return
    #print('There are now',time_spent_in_color_vst[0].shape[0],'pixels in reg',0)
    #print('There are now',time_spent_in_color_vst[1].shape[0],'pixels in reg',1)
    #print('There are now',time_spent_in_color_vst[2].shape[0],'pixels in reg',2)


def nth_true_element(a, toget):
    '''
    From array [a] of shape (d1 -- pixels, d2 -- colors) , get the [toget]'th True element along dimension d2.
    Outputs array of dim d1.
    '''
    cols_toadd_all = np.where(a)
    inds_newpixel = np.where(np.hstack(([1], np.diff(cols_toadd_all[0]))))[0]
    return cols_toadd_all[1][ inds_newpixel + toget ]

def show_progress(timefrac, timeflag, frequency=0.1):
    if timefrac > timeflag * frequency:
        timeflag += 1
        print('Ran {:.2f}% of the steps'.format(100 * timefrac), end='\r')
    return 1


def calc_stable_timefrac(cpart, t_step, t_lims,
                         time_spent_in_color, stable_colors):
        '''
        Compute the fraction of time that each pixel was in each color.
        The output stable_timefrac is of shape (n_pixels, n_colors), containing time fractions
        time_spent_in_color is of shape (n_pixels, n_colors), containing spent times (within the given timestep)
        stable_colors is of shape (n_pixels, n_colors), containing color indices ordered in decreasing occupation times
        '''

        res = np.take_along_axis(time_spent_in_color, stable_colors, axis=1)

        # Normalize by the total time the canvas section was on.
        for e in range(var.N_ENLARGE):
            res[cpart.canvassection_coordinds[e]] /= t_lims[t_step] - max(t_lims[t_step-1], var.TIME_ENLARGE[e])

        return res


#@profile
def main_variables(cpart,
                   cpst,
                   start_pixels=None,
                   delete_dir=False,
                   print_progress=False,
                   flattening='ravel',
                   compression='LZ77',
                   ref_im_const=None,
                   save_memory=None
                   ):
    '''
    Calculates the main variables that are contained in a CanvasPartStatistics object.
    This includes stability and attack/defense related variables.

    parameters
    ----------
    cpart : CanvasPart object
        The CanvasPart object for which we want to calculate the stability
    cpst : CanvasPartStatistics object
        The object whose attributes will receive all computed variables
    start_pixels : 1d array, optional, size (n_pixels)
        1D pixels to start with if t0!=0
    delete_dir : boolean, optional
        Remove directory after running, if it did not exist before
    print_progress : boolean, optional
        If True, prints a progress statement
    ref_im_const : None or 2d array
        If None, then the reference image used is calculated from the sliding window. If array, then the reference image
        is the array passed for this variable.
    save_memory : boolean (by default, it is None and then initialized within the function)
        If True, an approach about 2.5x slower is used to store time_spent_in_color using much less RAM

    returns
    -------
    returns nothing, but modifies many attributes of the input cpst
    '''

    # Preliminaries
    t_lims = cpst.t_lims
    n_tlims = len(t_lims)
    tmin = cpst.tmin
    itmin = np.argmax(t_lims >= tmin)
    attdef = cpst.compute_vars['attackdefense']
    stab = cpst.compute_vars['stability']
    instant = cpst.compute_vars['entropy']
    tran = cpst.compute_vars['transitions']
    other = cpst.compute_vars['other']
    ews = cpst.compute_vars['ews']
    inout = cpst.compute_vars['inoutgroup']
    lifetm = cpst.compute_vars['lifetime_vars']
    void_attack = cpst.compute_vars['void_attack']
    cluster = cpst.compute_vars['clustering']
    if save_memory is None:
        save_memory = (cpart.num_pix() > 3e5) # do the slower memory-saving method when the canvaspart is larger than 300,000 pixels

    # Some warnings for required conditions of use of the function
    if (not np.isclose(t_lims[0], cpart.min_max_time()[0], atol=1e-4)) and start_pixels is None: # TODO: check that we both agree to adding atlas_min=True here
        warnings.warn('t_lims parameter should start with the minimum time of the CanvasPart, or the start_pixels should be provided, for standard functioning.')
    if delete_dir and (instant > 2 or stab > 2 or attdef > 2):
        warnings.warn('Some images were required to be stored, but with delete_dir=True, the full directory will be removed! ')

    # Canvas part data
    cpart.set_canvassection_coordinds() 
    coor_offset = cpart.coords_offset()  # for transformation from 1D to 2D pixels

    # Initialize variables for first time iteration
    if ref_im_const is None:
        ref_color = cpart.white_image(1)  # reference image (stable over the sliding window)
    else:
        ref_color = ref_im_const
    current_color = cpart.white_image(1) if start_pixels is None else start_pixels
    if instant > 0 or tran > 0:
        previous_colors = cpart.white_image(2, images_number=cpst.sw_width) # image in the sw_width previous timesteps
    if cluster > 0:
        previous_downscaled_im = None
    previous_stable_color = cpart.white_image(1)  # stable image in the previous timestep
    if save_memory:
        active_colors_vst = np.zeros((cpst.sw_width, cpart.num_pix(), var.NUM_COLORS), dtype=bool)
        rec_lengths = np.array([int(cpst.sw_width * 1.7), int(cpst.sw_width * 4.5), cpst.sw_width * var.NUM_COLORS])
        rec_len_regime = np.zeros((cpart.num_pix()), dtype=np.int8)
        rec_timespent_num = np.zeros((cpart.num_pix()), dtype=np.int16)
        time_spent_in_color_vst = np.empty((3), dtype=object)
        time_spent_in_color_vst[0] = np.zeros((cpart.num_pix(), rec_lengths[0]), dtype=np.float32)
        for i in np.arange(1, 3): # will be filled when necessary
            time_spent_in_color_vst[i] = np.zeros((0, rec_lengths[i]), dtype=np.float32)
        pos_in_timespentvst = np.arange(cpart.num_pix(), dtype=np.int32)
        startidx_timespent = np.zeros((cpart.num_pix()), dtype=np.int32)
        endidx_timespent = np.zeros((cpart.num_pix()), dtype=np.int32)

    else:
        time_spent_in_color_vst = np.zeros((cpst.sw_width, cpart.num_pix(), var.NUM_COLORS))
    time_spent_in_color_sw = np.zeros((cpart.num_pix(), var.NUM_COLORS)) # summed over the sliding window
    last_time_installed_sw = None if attdef == 0 else initialize_start_time_grid(cpart, t_lims[0], add_color_dim=True)
    last_time_removed_sw = None if attdef == 0 else np.copy(last_time_installed_sw)
    users_vst = np.empty(cpst.sw_width, dtype=object)
    for i in np.ndindex(users_vst.shape):
        users_vst[i] = np.array([], dtype=np.int32)
    users_sw_unique = np.array([], dtype=np.int32)
    stable_colors = np.empty((cpart.num_pix(), var.NUM_COLORS), dtype=np.int8)
    if ews > 0:
        stable_colors_prev = np.empty((cpart.num_pix(), var.NUM_COLORS), dtype=np.int8)
    stable_timefrac_prev = np.empty((cpart.num_pix(), var.NUM_COLORS))

    # Output
    cpst.stability = np.empty(4, dtype=object)
    cpst.runnerup_timeratio = np.empty(4, dtype=object)
    cpst.n_used_colors = np.empty(4, dtype=object)
    for i in range(0, 4):
        cpst.stability[i] = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
        cpst.runnerup_timeratio[i] = cpst.ts_init(np.zeros(n_tlims, dtype=np.float32))
        cpst.n_used_colors[i] = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
    cpst.autocorr_bycase = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.autocorr_bycase_norm = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.autocorr_multinom = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.autocorr_subdom = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.autocorr_dissimil = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    #cpst.variance = cpst.ts_init(np.ones(n_tlims, dtype=np.float64))
    cpst.variance2 = cpst.ts_init(np.ones(n_tlims, dtype=np.float64))
    cpst.variance_multinom = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.variance_subdom = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
    cpst.diff_pixels_stable_vs_swref = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_swref = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_swref_forwardlook = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_inst = cpst.ts_init(np.zeros(n_tlims))
    cpst.diff_pixels_inst_vs_stable = cpst.ts_init(np.zeros(n_tlims))
    if attdef > 0:
        cpst.returnrate = cpst.ts_init(np.zeros(n_tlims, dtype=np.float64))
        cpst.returntime = np.empty(4, dtype=object)
        for i in range(0, 4):
            cpst.returntime[i] = cpst.ts_init(np.zeros(n_tlims, dtype=np.float32))
    cpst.area_vst = cpst.ts_init(np.full(n_tlims, cpart.num_pix()))
    cpst.n_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_defense_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_bothattdef_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_defenseonly_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_attackonly_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_ingroup_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_outgroup_changes = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_ingrouponly_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_outgrouponly_users = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_bothinout_users = cpst.ts_init(np.zeros(n_tlims))
    if inout > 1:
        cpst.ingroup_users_vst = np.empty(n_tlims, dtype=object)
        cpst.outgroup_users_vst = np.empty(n_tlims, dtype=object)
        for j in range(n_tlims):
            cpst.ingroup_users_vst[j] = np.array([], dtype=np.int32)
            cpst.outgroup_users_vst[j] = np.array([], dtype=np.int32)
    cpst.num_edge_pixels = cpst.ts_init(np.zeros(n_tlims))
    cpst.frac_attack_changes_image = np.full((n_tlims, cpart.width(1), cpart.width(0)), 1, dtype=np.float16) if attdef > 1 else None
    cpst.size_uncompressed = cpst.ts_init(np.zeros(n_tlims))
    cpst.size_compressed = cpst.ts_init(np.zeros(n_tlims))
    cpst.size_compr_stab_im = cpst.ts_init(np.zeros(n_tlims))
    cpst.cumul_attack_timefrac = cpst.ts_init(np.zeros(n_tlims))
    cpst.n_moderator_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_cooldowncheat_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_redundant_color_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_redundant_coloranduser_changes = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_new_vs_previoustime = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_new_vs_sw = cpst.ts_init( np.zeros(n_tlims) )
    cpst.n_users_sw = cpst.ts_init( np.zeros(n_tlims) )
    cpst.fractal_dim_weighted = cpst.ts_init( np.full(n_tlims, 2) )
    cpst.fractal_dim_mask_median = cpst.ts_init( np.full(n_tlims, 2) )
    cpst.complexity_multiscale = cpst.ts_init( np.zeros(n_tlims) )
    cpst.complexity_levenshtein = cpst.ts_init( np.zeros(n_tlims) )
    cpst.diff_pixels_inst_vs_inst_downscaled2 = cpst.ts_init( np.zeros(n_tlims) )
    cpst.diff_pixels_inst_vs_inst_downscaled4 = cpst.ts_init( np.zeros(n_tlims) )
    cpst.diff_pixels_inst_vs_inst_downscaled16pix = cpst.ts_init( np.zeros(n_tlims) )
    cpst.maxscale = None
    cpst.wavelet_high_freq = cpst.ts_init( np.zeros(n_tlims) )
    cpst.wavelet_mid_freq = cpst.ts_init( np.zeros(n_tlims) )
    cpst.wavelet_low_freq = cpst.ts_init( np.zeros(n_tlims) )
    cpst.wavelet_low_freq_tm = cpst.ts_init( np.zeros(n_tlims) )
    cpst.wavelet_high_freq_tm = cpst.ts_init( np.zeros(n_tlims) )
    cpst.frac_black_px = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
    cpst.frac_purple_px = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
    cpst.frac_black_ref = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
    cpst.frac_purple_ref = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
    attack_users = np.array([])
    defense_users = np.array([])
    bothattdef_users = np.array([])
    ingroup_users = np.array([])
    outgroup_users = np.array([])
    bothinout_users = np.array([])
    outgroup_inds = np.array([], dtype=np.int32)
    if cluster > 0:
        cpst.ripley = np.empty(len(cpst.ripley_distances), dtype=object)
        cpst.ripley_norm = np.empty(len(cpst.ripley_distances), dtype=object)
        for i in range(0, len(cpst.ripley_distances)):
            cpst.ripley[i] = cpst.ts_init(np.zeros(n_tlims, dtype=np.float32))
            cpst.ripley_norm[i] = cpst.ts_init(np.zeros(n_tlims, dtype=np.float32))
        cpst.crosscorr = np.empty(len(cpst.crosscorr_distances), dtype=object)
        for i in range(0, len(cpst.crosscorr_distances)):
            cpst.crosscorr[i] = cpst.ts_init(np.zeros(n_tlims, dtype=np.float32))
        cpst.dist_average = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
        cpst.dist_average_norm = cpst.ts_init(np.ones(n_tlims, dtype=np.float32))
        cpst.ripley_ref = None
        cpst.avdist_ref = None



    # output paths
    out_path = os.path.join(var.FIGS_PATH, cpart.out_name())
    dir_exist_already = util.make_dir(out_path, renew=False)
    out_dir_time = 'VsTimeInst'
    out_dir_ref = 'reference_image'
    out_dir_stab = 'VsTimeStab'
    out_dir_attdef = 'attack_defense_ratio'
    out_dir_attdefIsing = 'attack_defense_Ising'
    cpart_dir = lambda d : os.path.join(cpart.out_name(), d)
    fig_cpart_dir = lambda d : os.path.join(var.FIGS_PATH, cpart_dir(d))

    # Initialize stored pixels and images
    if stab > 1:
        # 2d numpy arrays containing color indices (from 0 to 31) for each pixel of the composition
        cpst.stable_image = cpart.white_image(3, images_number=n_tlims)
        cpst.second_stable_image = cpart.white_image(3, images_number=n_tlims)
        cpst.third_stable_image = cpart.white_image(3, images_number=n_tlims)

        def modify_some_pixels(start, target, step, indices):
            ''' start and target are 1d arrays of 2d images.
            Coordinates at given indices in [start] must be replaced by the content of [target]'''
            start[step, coor_offset[1, indices], coor_offset[0, indices]] = target[step, coor_offset[1, indices], coor_offset[0, indices]]

        if stab > 2:
            util.make_dir(fig_cpart_dir(out_dir_stab), renew=True)

    if tran > 0:
        cpst.refimage_sw_flat = cpart.white_image(2, images_number=n_tlims)

    if attdef > 1 or tran > 1:
        cpst.refimage_sw = cpart.white_image(3, images_number=n_tlims)
        cpst.attack_defense_image = cpart.white_image(3, images_number=n_tlims)
        if attdef > 2:
            util.make_dir(fig_cpart_dir(out_dir_attdef), renew=True)
            util.make_dir(fig_cpart_dir(out_dir_attdefIsing), renew=True)

    if instant > 0 or tran > 1:
        cpst.true_image = cpart.white_image(3, images_number=n_tlims)
    if instant > 2:
        util.make_dir(fig_cpart_dir(out_dir_time), renew=True)

    # Start with a white image for time 0
    if ((compression == 'DEFLATE_BMP_PNG') or (instant > 2)):
        if (compression == 'DEFLATE_BMP_PNG') and (flattening != 'ravel'):
            warnings.warn(('Compression algorithm DEFLATE with BMP to PNG can only handle ravel flattening. Using ravel'
                            ' instead of ' + flattening))
        create_files_and_get_sizes(t_lims, 0, cpart.white_image(2),
                                   cpst.size_compressed.val, cpst.size_uncompressed.val, cpart_dir(out_dir_time))
    
    # For identifying in/outgroup by comparing to sliding window looking forward and backward. Lists stay empty if inout is 0.
    agreeing_changes_vst = []
    t_inds_active_vst = []

    # image with radial distance from the center, for cross-correlation computation
    if cluster > 0:
        dummy_image = cpart.white_image(2).astype(np.float32)
        H, W = fftconvolve(dummy_image, dummy_image, mode='full').shape
        y, x = np.indices((H, W))
        image_radius_flat = np.round(np.sqrt((y - H//2)**2 + (x - W//2)**2)).astype(int).ravel()
        del y, x, dummy_image, H, W
        crosscorr_bins = np.append(cpst.crosscorr_distances, 10000)
        crosscorr_radial_ref, _ = np.histogram(image_radius_flat, bins=crosscorr_bins)

    # LOOP over time steps
    i_fraction_print = 0
    for i in range(1, n_tlims):
        # Print output showing fraction of the steps that have run.
        if print_progress:
            show_progress(i/n_tlims, i_fraction_print, 0.1)

        timerange_str = 'time{:06d}to{:06d}'.format(int(t_lims[i-1]), int(t_lims[i]))

        # Get indices of all pixels that are active in this time step.
        inds_coor_active = cpart.active_coord_inds(t_lims[i-1], t_lims[i])
        cpst.area_vst.val[i] = len(inds_coor_active)
        # Get indices of pixel changes in this time step, without or with the condition of being in an "active" pixel at this time
        t_inds = cpart.intimerange_pixchanges_inds(t_lims[i-1], t_lims[i])
        t_inds_active = cpart.select_active_pixchanges_inds(t_inds)
        if inout > 0:
            t_inds_active_vst.append(t_inds_active)

        # CORE COMPUTATIONS. Magic happens here
        # Misc. pixel changes variables
        if other > 0 and i >= itmin - 1:
            cpst.n_moderator_changes.val[i] = np.count_nonzero(cpart.moderator(t_inds_active))
            cpst.n_cooldowncheat_changes.val[i] = np.count_nonzero(cpart.cheat(t_inds_active))
            cpst.n_redundant_color_changes.val[i] = np.count_nonzero(cpart.redundant(t_inds_active))
            cpst.n_redundant_coloranduser_changes.val[i] = np.count_nonzero(cpart.superredundant(t_inds_active))

        # Store the image of the previous timestep
        if stab > 0 and instant > 0 and i >= itmin - 1:
            previous_stable_color = np.copy(stable_colors[:, 0]) if i > 1 else cpart.white_image(1)

        # Calculate the time each pixel spent in what color.
        # This also updates [current_color, last_time_installed_sw, last_time_removed_sw] with the pixel changes in this time step
        # time_spent_in_color is of shape (n_pixels, n_colors)
        time_spent_in_color = calc_time_spent_in_color(cpart, t_inds,
                                                       t_lims[i-1], t_lims[i], current_color,
                                                       last_time_installed_sw, last_time_removed_sw)

        # STABILITY
        if stab > 0 and i >= itmin - 1:
            # Get the color indices in descending order of which color they spent the most time in
            # stable_colors is of shape (n_pixels, n_colors), containing color indices ordered in decreasing occupation times
            stable_colors = calc_stable_cols(time_spent_in_color)
            # Get the times spent in each color in descending order of time spent.
            # stable_timefrac is of shape (n_pixels, n_colors), containing time fractions
            stable_timefrac = calc_stable_timefrac(cpart, i, t_lims,
                                                   time_spent_in_color, stable_colors)
            # calculate the stability value in the time interval
            stabil, runnerup, ncols = calc_stability(stable_timefrac, inds_coor_active, True)
            for k in range(0, 4):
                cpst.stability[k].val[i] = stabil[k]
                cpst.runnerup_timeratio[k].val[i] = runnerup[k]
                cpst.n_used_colors[k].val[i] = ncols[k]

        if stab > 0 or tran > 0 or attdef > 0:
            # Store time_spent_in_color for this timestep, in a sparse array
            i_replace = i % cpst.sw_width # where to modify the "rolling" time_spent_in_color_vst array
            if save_memory:
                substract_and_sparse_replace(time_spent_in_color_sw, time_spent_in_color_vst, time_spent_in_color, i_replace,
                                             pos_in_timespentvst, active_colors_vst, startidx_timespent, endidx_timespent,
                                             rec_len_regime, rec_lengths, rec_timespent_num, cpst.sw_width, try_reduce=(i % (4*cpst.sw_width) == 0))
            else:
                time_spent_in_color_sw -= time_spent_in_color_vst[i_replace]
                time_spent_in_color_vst[i_replace] = np.copy(time_spent_in_color)
            time_spent_in_color_sw += time_spent_in_color

        # Calculate the new reference image
        if ref_im_const is None:
            ref_color = calc_stable_cols(time_spent_in_color_sw)[:, 0]
        else:
            ref_color = ref_im_const

        if i < itmin - 1: # skip times where the composition is not on
            continue

        # CLASSIC EWS VARIABLES
        if ews > 0:
            compute_classic_ews(cpst, i, inds_coor_active, 
                                stable_timefrac, stable_timefrac_prev,
                                stable_colors, stable_colors_prev, 
                                previous_colors, current_color )
            stable_colors_prev = np.copy(stable_colors)
            
        # ATTACK/DEFENSE VS REFERENCE IMAGE
        if attdef > 0:
            # Calculate the (normalized) cumulative attack time over all pixels in this timestep
            cpst.cumul_attack_timefrac.val[i] = cumulative_attack_timefrac(time_spent_in_color, ref_color,
                                                                           inds_coor_active, cpst.t_interval)

            cpst.returnrate.val[i] = returnrate(current_color, previous_colors[(i - 1) % cpst.sw_width],
                                                ref_color, inds_coor_active, last_time_installed_sw, t_lims, i)
            returnt = returntime(last_time_installed_sw, last_time_removed_sw,
                                current_color, ref_color,
                                inds_coor_active, t_lims[i], cpst.sw_width_sec)
            for k in range(0, 4):
                cpst.returntime[k].val[i] = returnt[k]

            # Ising-style map of attack vs defense pixels
            if attdef > 2:
                diff_image = current_color - ref_color
                diff_image_isref = (diff_image == 0)
                diff_image[diff_image_isref] = 31  # white if defense
                diff_image[np.invert(diff_image_isref)] = 5  # black if attack
                cpst.attack_defense_image[i][coor_offset[1, inds_coor_active],
                                             coor_offset[0, inds_coor_active]] = diff_image[inds_coor_active]

            # Calculate the number of changes and of users that are attacking or defending the reference image. Keep users_sw_unique for the next step
            n_changes_and_users_result = num_changes_and_users(cpart, cpst, i, i_replace, timerange_str,
                                                    t_inds_active, ref_color, users_vst, users_sw_unique,
                                                    t_inds_active_vst,
                                                    agreeing_changes_vst,
                                                    attdef > 0, inout > 0, lifetm > 0, attdef > 1,
                                                    (cpart_dir(out_dir_attdef) if attdef > 2 else ''))
            users_sw_unique = n_changes_and_users_result[0]
            if lifetm > 0:
                if attdef > 0:
                    attack_users = np.concatenate((attack_users, np.atleast_1d(n_changes_and_users_result[1])))
                    defense_users = np.concatenate((defense_users, np.atleast_1d(n_changes_and_users_result[2])))
                    bothattdef_users = np.concatenate((bothattdef_users, np.atleast_1d(n_changes_and_users_result[3])))
                if inout > 0 and i >= cpst.sw_width:
                    ingroup_users = np.concatenate((ingroup_users, np.atleast_1d(n_changes_and_users_result[4])))
                    outgroup_users = np.concatenate((outgroup_users, np.atleast_1d(n_changes_and_users_result[5])))
                    bothinout_users = np.concatenate((bothinout_users, np.atleast_1d(n_changes_and_users_result[6])))
                    outgroup_inds = np.concatenate((outgroup_inds, np.atleast_1d(n_changes_and_users_result[7])))


        # INSTANTANEOUS IMAGES: includes entropy and fractal dimension calculations
        # Calculate the number of pixels in the current interval (stable or instantaneous) that differ from the reference image, or from the previous timestep
        if tran > 0:
            cpst.diff_pixels_inst_vs_swref.val[i] = np.count_nonzero(current_color[inds_coor_active] - ref_color[inds_coor_active])
            if i >= cpst.sw_width:
                cpst.diff_pixels_inst_vs_swref_forwardlook.val[i - cpst.sw_width] = np.count_nonzero(previous_colors[i_replace, inds_coor_active] - ref_color[inds_coor_active])
        if stab > 0 and instant > 0:
            cpst.diff_pixels_stable_vs_swref.val[i] = np.count_nonzero(stable_colors[inds_coor_active, 0] - ref_color[inds_coor_active])
            cpst.diff_pixels_inst_vs_stable.val[i] = np.count_nonzero(current_color[inds_coor_active] - previous_stable_color[inds_coor_active])
        if instant > 0 or tran > 1:
            # ENTROPY AND OTHER COMPLEXITY METRICS
            cpst.diff_pixels_inst_vs_inst.val[i] = np.count_nonzero(current_color[inds_coor_active] - previous_colors[(i-1) % cpst.sw_width, inds_coor_active])

            # Create the png and bmp files from the current image, and store their sizes
            pix_tmp = cpart.white_image(2)
            pix_tmp[coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = current_color[inds_coor_active]
            if void_attack > 0:
                cpst.frac_black_px.val[i] = len(np.where(current_color[inds_coor_active] == var.BLACK)[0])/len(current_color[inds_coor_active])
                cpst.frac_purple_px.val[i] = len(np.where(current_color[inds_coor_active] == var.PURPLE)[0])/len(current_color[inds_coor_active])
                cpst.frac_black_ref.val[i] = len(np.where(ref_color[inds_coor_active] == var.BLACK)[0])/len(ref_color[inds_coor_active])
                cpst.frac_purple_ref.val[i] = len(np.where(ref_color[inds_coor_active] == var.PURPLE)[0])/len(ref_color[inds_coor_active])
            if ((compression == 'DEFLATE_BMP_PNG') or (instant > 2)):
                create_files_and_get_sizes(t_lims, i, pix_tmp,
                                           cpst.size_compressed.val, cpst.size_uncompressed.val,
                                           cpart_dir(out_dir_time), True, instant < 3)
            else:
                cpst.size_compressed.val[i] = entropy.calc_compressed_size(pix_tmp, flattening=flattening, compression=compression)
                cpst.size_uncompressed.val[i] = entropy.calc_size(pix_tmp)
            
            # Wavelet analysis for complexity metric
            cpst.wavelet_low_freq.val[i], cpst.wavelet_mid_freq.val[i], cpst.wavelet_high_freq.val[i] = entropy.compute_wavelet_energies(pix_tmp)

            # Wavelet time series
            cpst.wavelet_low_freq_tm.val[i], cpst.wavelet_high_freq_tm.val[i] = entropy.compute_wavelet_energies_time(cpst.diff_pixels_inst_vs_swref.val[max(0, i-cpst.sw_width):i])

            # Multiscale complexity
            scales = None if cpst.maxscale is None else [2 ** i for i in range(0, int(np.log2(cpst.maxscale)+0.1)+1)]
            scales, downscaled_im = entropy.downsampled_images(pix_tmp, scales=scales)
            if cpst.maxscale is None:
                cpst.maxscale = scales[-1]
            cpst.complexity_multiscale.val[i] = entropy.compute_complexity_multiscale(pix_tmp, scales, downscaled_im)

            # fraction of differing pixels with downscaled images
            if previous_downscaled_im is None:
                previous_downscaled_im = copy.deepcopy(downscaled_im)
            cpst.diff_pixels_inst_vs_inst_downscaled2.val[i] = np.count_nonzero(downscaled_im[1]-previous_downscaled_im[1]) if len(downscaled_im) > 1 else 0
            cpst.diff_pixels_inst_vs_inst_downscaled4.val[i] = np.count_nonzero(downscaled_im[2]-previous_downscaled_im[2]) if len(downscaled_im) > 2 else cpst.diff_pixels_inst_vs_inst_downscaled2.val[i]
            cpst.diff_pixels_inst_vs_inst_downscaled16pix.val[i] = np.count_nonzero(downscaled_im[-1]-previous_downscaled_im[-1]) if len(downscaled_im) > 1 else 0
            previous_downscaled_im = copy.deepcopy(downscaled_im)

            # Levenshtein complexity
            cpst.complexity_levenshtein.val[i] = entropy.compute_complexity_levenshtein(pix_tmp)

        # create instantaneous images.
        if instant > 0 or tran > 1:
            cpst.true_image[i] = np.copy(pix_tmp)
            del pix_tmp

        if cluster > 0:
            # Ripley's K function and average distance between pixel changes
            # When must the reference values (using random locations of pixels) be calculated? At t=0 or when the borders changed
            # meaning whenever the current time is less than a time step away from the start of a cpst.stable_borders_timeranges
            change_of_borders = np.any((t_lims[i] < cpst.stable_borders_timeranges[:, 0] + cpst.t_interval) & (t_lims[i] >= cpst.stable_borders_timeranges[:, 0]))
            if change_of_borders:
                cpst.ripley_ref = None
                cpst.avdist_ref = None

            # actual calculation
            calc_changes_clustering(cpst, i, 
                                    cpart.pixchanges_coords_offset(t_inds_active), coor_offset[:, inds_coor_active])

            # cross correlation
            cross_correlation(cpst, i, itmin, image_radius_flat, crosscorr_bins, crosscorr_radial_ref)
            
        if tran > 0 or instant > 0:
            previous_colors[i_replace] = np.copy(current_color)

        # END CORE COMPUTATIONS. Magic ends

        # Save full result to pickle file.
        if stab > 3:
            file_path = os.path.join(var.DATA_PATH, 'stability_' + cpart.out_name() + '_time{:06d}to{:06d}.pickle'.format(int(t_lims[i-1]), int(t_lims[i])))
            with open(file_path, 'wb') as handle:
                pickle.dump([stable_timefrac,
                            stable_colors],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # Create attack/defense images.
        if tran > 0:
            cpst.refimage_sw_flat[i] = ref_color
        if attdef > 1 or tran > 1:
            cpst.refimage_sw[i, coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = ref_color[inds_coor_active]
            if attdef > 2:
                timerange_str_ref = 'time{:06d}to{:06d}'.format(int(t_lims[i] - cpst.sw_width), int(t_lims[i]))
                util.pixels_to_image(cpst.refimage_sw[i], cpart_dir(out_dir_ref), 'SlidingRef_' + timerange_str_ref + '.png')
                util.pixels_to_image(cpst.attack_defense_image[i], cpart_dir(out_dir_attdefIsing), 'Attack_vs_defense_' + timerange_str + '.png')
        del ref_color
        # Create images containing the (sub)dominant color only.
        if stab > 1 and i >= itmin:
            cpst.stable_image[i, coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = stable_colors[inds_coor_active, 0]
            cpst.second_stable_image[i, coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = stable_colors[inds_coor_active, 1]
            cpst.third_stable_image[i, coor_offset[1, inds_coor_active], coor_offset[0, inds_coor_active]] = stable_colors[inds_coor_active, 2]

            # Calculate entropy of most stable image
            cpst.size_compr_stab_im.val[i] = entropy.calc_compressed_size(cpst.stable_image[i], flattening=flattening, compression=compression)

            # If second and/or third most used colors don't exist (time_spent == 0),
            # then use the first or second most used color instead.
            inds_to_change1 = np.where(stable_timefrac[:, 1] < 1e-9)
            modify_some_pixels(cpst.second_stable_image, cpst.stable_image, i, inds_to_change1)
            inds_to_change2 = np.where(stable_timefrac[:, 2] < 1e-9)
            modify_some_pixels(cpst.third_stable_image, cpst.second_stable_image, i, inds_to_change2)

            if stab > 2:
                # Save images.
                util.pixels_to_image(cpst.stable_image[i], cpart_dir(out_dir_stab), 'MostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(cpst.second_stable_image[i], cpart_dir(out_dir_stab), 'SecondMostStableColor_' + timerange_str + '.png')
                util.pixels_to_image(cpst.third_stable_image[i], cpart_dir(out_dir_stab), 'ThirdMostStableColor_' + timerange_str + '.png')
            
        del inds_coor_active
        if stab > 0:
            del stable_timefrac
        del time_spent_in_color

    
    # FRACTAL DIMENSION AND OTHER COMPLEXITY MEASURES
    if instant > 0:
        [cpst.fractal_dim_mask_median.val,
         cpst.fractal_dim_weighted.val] = fractal_dim.calc_from_image(cpst, shift_avg=True)

    # Continue the loop over some more time steps to get the forward-looking sliding window reference
    if tran > 0:
        for i in range(n_tlims, n_tlims + min(cpst.sw_width, n_tlims-1)):
            if n_tlims - 1 >= cpst.sw_width:
                back_index = i - 1 - cpst.sw_width
            else:
                back_index = i - n_tlims
        
            inds_coor_active = cpart.active_coord_inds(t_lims[back_index], t_lims[back_index + 1])

            i_replace = i % cpst.sw_width # where to modify the "rolling" time_spent_in_color_vst array
            if save_memory:
                substract_and_sparse_replace(time_spent_in_color_sw, time_spent_in_color_vst, None, i_replace,
                                             pos_in_timespentvst, active_colors_vst, startidx_timespent, None,
                                             rec_len_regime, rec_lengths, None, None, replace=False, try_reduce=False)
            else:
                # time_spent_in_color_vst has length of the number of intervals in the sw (cpst.sw_width)
                # it contains the time spent in each color and pixel for each interval in the SW
                # time_spent_in_color_sw is the total time spent in each color and pixel for each interval 
                time_spent_in_color_sw -= time_spent_in_color_vst[i_replace]
            
            ref_color = calc_stable_cols(time_spent_in_color_sw)[:, 0]

            if inout > 0:
                # t_inds_active is length of the t_intervals
                t_inds_active_fwd = t_inds_active_vst[back_index]

                agreeing_changes_bkwd = agreeing_changes_vst[back_index]
                # Check whether there are enough time steps after the back index to at least go the edge sliding width forward in time 
                if back_index + cpst.sw_width_edge <= n_tlims-1:
                    agreeing_changes_fwd = np.array(ref_color[cpart.coordidx(t_inds_active_fwd)] == cpart.color(t_inds_active_fwd), np.bool_)
                    ingroup_changes = agreeing_changes_bkwd | agreeing_changes_fwd
                else:
                    ingroup_changes = agreeing_changes_bkwd # backward or forward
                outgroup_changes = np.invert(ingroup_changes)
                cpst.n_ingroup_changes.val[back_index+1] = np.count_nonzero(ingroup_changes)
                cpst.n_outgroup_changes.val[back_index+1] = np.count_nonzero(outgroup_changes)

                # users
                ingroup_users_step = np.unique(cpart.user(t_inds_active_fwd)[ingroup_changes])
                outgroup_users_step = np.unique(cpart.user(t_inds_active_fwd)[outgroup_changes])
                bothinout_users_step = np.intersect1d(ingroup_users_step, outgroup_users_step)
                ingroup_users = np.concatenate((ingroup_users, np.atleast_1d(ingroup_users_step)))
                outgroup_users = np.concatenate((outgroup_users, np.atleast_1d(outgroup_users_step)))
                bothinout_users = np.concatenate((bothinout_users, np.atleast_1d(bothinout_users_step)))
                cpst.n_bothinout_users.val[back_index+1] = len(bothinout_users_step)
                cpst.n_outgrouponly_users.val[back_index+1] = len(outgroup_users_step) - len(bothinout_users_step)
                cpst.n_ingrouponly_users.val[back_index+1] = len(ingroup_users_step) - len(bothinout_users_step)
                if cpst.ingroup_users_vst is not None:
                    cpst.ingroup_users_vst[back_index+1] = np.copy(ingroup_users_step)
                    cpst.outgroup_users_vst[back_index+1] = np.copy(outgroup_users_step)

                if inout > 1 and (attdef > 1 or tran > 1):
                    edges = label_edge_map(cpst.refimage_sw[back_index])
                    cpst.num_edge_pixels.val[back_index] = np.count_nonzero(edges)


            cpst.diff_pixels_inst_vs_swref_forwardlook.val[back_index+1] = np.count_nonzero(previous_colors[i_replace, inds_coor_active] - ref_color[inds_coor_active])

    # LIFETIME VALUES
    # calculate some things after the loop to get the value over the lifetime of the composition
    if lifetm > 0:
        if attdef > 0:
            n_intersect_att_def = len(np.unique(np.intersect1d(attack_users, defense_users)))
            cpst.n_attackonly_users_lifetime = len(np.unique(attack_users)) - n_intersect_att_def
            cpst.n_defenseonly_users_lifetime = len(np.unique(defense_users)) - n_intersect_att_def
            cpst.n_bothattdef_users_lifetime = len(np.unique(bothattdef_users)) + n_intersect_att_def
        if inout > 0:
            n_intersect_in_out = len(np.unique(np.intersect1d(outgroup_users, ingroup_users)))
            cpst.n_outgrouponly_users_lifetime = len(np.unique(outgroup_users)) - n_intersect_in_out
            cpst.n_ingrouponly_users_lifetime = len(np.unique(ingroup_users)) - n_intersect_in_out
            cpst.n_bothinout_users_lifetime = len(np.unique(bothinout_users)) + n_intersect_in_out
            cpst.outgroup_inds = outgroup_inds

    # These calculations can be done on the final arrays
    inds_active_all = inds_active_all = np.where((cpart.pixel_changes['active']==True) & (cpart.pixel_changes['seconds'] < cpst.tmax))[0]
    cpst.n_users_total = len(np.unique(cpart.pixel_changes['user'][inds_active_all]))

    if print_progress:
        print('                              ', end='\r')

    if delete_dir and not dir_exist_already:
        os.rmdir(out_path)

def compute_classic_ews(cpst, i, inds_coor_active, 
                        stable_timefrac, stable_timefrac_prev,
                        stable_colors, stable_colors_prev, 
                        previous_colors, current_color):

    mode_color = stable_colors[inds_coor_active, 0]
    prev_color = previous_colors[(i - 1) % cpst.sw_width, inds_coor_active]
    curr_color = current_color[inds_coor_active]

    # autocorrelation averaged over pixels
    autocorr_norm_per_pix = np.zeros(len(curr_color))
    autocorr_norm_per_pix[(curr_color == prev_color) & (curr_color == mode_color)] = 0
    autocorr_norm_per_pix[(curr_color == prev_color) & (curr_color != mode_color)] = 1
    autocorr_norm_per_pix[(curr_color != prev_color) & (curr_color != mode_color) & (prev_color != mode_color)] = -1
    autocorr_norm_per_pix[(curr_color != prev_color) & ((curr_color == mode_color) | (prev_color == mode_color))] = 0
    cpst.autocorr_bycase.val[i] = np.mean(autocorr_norm_per_pix) if len(autocorr_norm_per_pix) > 0 else 0

    # autocorrelation normalized on entire state
    autocorr_denom_per_pix = np.zeros(len(curr_color))
    autocorr_denom_per_pix[(curr_color == prev_color) & (curr_color == mode_color)] = 0
    autocorr_denom_per_pix[(curr_color == prev_color) & (curr_color != mode_color)] = 1
    autocorr_denom_per_pix[(curr_color != prev_color) & (curr_color != mode_color) & (prev_color != mode_color)] = 1
    autocorr_denom_per_pix[(curr_color != prev_color) & ((curr_color == mode_color) | (prev_color == mode_color))] = 0.5
    denom = np.sum(autocorr_denom_per_pix)
    cpst.autocorr_bycase_norm.val[i] = np.sum(autocorr_norm_per_pix) / denom if denom != 0 else 0

    timefrac = stable_timefrac[inds_coor_active]
    timefrac_prev = stable_timefrac_prev[inds_coor_active]
    # get stable_timefrac in the standard color order
    if i > 1:
        indspix = np.arange(len(inds_coor_active))
        timefrac[indspix[:,np.newaxis], stable_colors[indspix, :]] = stable_timefrac[inds_coor_active]
        timefrac_prev[indspix[:,np.newaxis], stable_colors_prev[indspix, :]] = stable_timefrac_prev[inds_coor_active]


    ## variance normalized per pixel (different definition from 1-stability)
    ## this sums the time fractions over each color for a given pixel, takes the inverse, then the mean over pixels
    #sum2_stable_timefrac = np.sum(timefrac**2, axis=1)
    #cpst.variance.val[i] = np.mean(np.reciprocal(sum2_stable_timefrac)) if len(inds_coor_active) > 0 else 1

    # variance of entire state
    # this sums the time fractions over each pixel, averaged over all colors, then takes the inverse
    cpst.variance2.val[i] = 1 / np.mean(timefrac**2) / var.NUM_COLORS if len(inds_coor_active) > 0 else 1

    # variance defined as for a multinomial distribution: sum_{color i}{ p_i(1-p_i) } = 1 - sum(p_i^2)
    cpst.variance_multinom.val[i] = np.mean( timefrac * (1-timefrac) ) * var.NUM_COLORS
    # autocorrelation defined in analogy with the variance of multinomial distribution: sum_{color i}{ p^t_i * (1-p_^{t-1}_i) }
    if i > 1:
        cpst.autocorr_multinom.val[i] = np.mean( timefrac_prev * (1-timefrac) ) * var.NUM_COLORS

        # autocorrelation based on chi^2 dissimilarity between the two distributions: sum_{color i}{ 0.5* (p^t_i - p_^{t-1}_i) ^2 }
        cpst.autocorr_dissimil.val[i] = 0.5 * np.mean( (timefrac_prev - timefrac)**2 ) * var.NUM_COLORS

    # variance and autocorrelation based on subdominant dot product. Sum of squares of X_i Y_i, excluding the color with the largest X_i Y_i term 
    subdom_sum = lambda a : np.mean( np.sum(a, axis=1) - np.max(a, axis=1) )
    if i > 1:
        cpst.autocorr_subdom.val[i] = subdom_sum( timefrac * timefrac_prev )
    cpst.variance_subdom.val[i] = subdom_sum( timefrac**2 )

    if i > 0:
        np.copyto(stable_timefrac_prev, stable_timefrac)

    return

def num_changes_and_users(cpart, cpst,
                          t_step, i_replace, time_str,
                          t_inds_active, ref_image, users_vst, users_sw_unique,
                          t_inds_active_vst,
                          agreeing_changes_vst,
                          calc_attdef_userlist, 
                          calc_inout_group,
                          calc_lifetime_vals,
                          save_ratio_pixels, 
                          save_ratio_images=''):
    """
    Analyzes and updates statistics related to pixel changes and user activities in a canvas part.

    This function updates several attributes of the `cpst` canvas_part_statistics object related to the number of pixel changes, 
    the classification of these changes (defense/attack), and user participation metrics. 
    It also handles the calculation of ratios for attack and defense changes at the pixel level and optionally saves these ratios.

    Parameters
    ----------
    cpart : CanvasPart object
        The canvas part object being analyzed.
    cpst : CanvasPartStatistics object
        Object containing statistical information of the canvas part, including counters and metrics to be updated.
    t_step : int
        The current time step index.
    i_replace : int
        Index for replacing users' list in a sliding window mechanism.
    time_str : str
        A string representation of the current time, used for naming saved files.
    t_inds_active : array_like
        Indices of active pixels in the current time step.
    ref_image : ndarray, shape (#y_coords, #x_coords)
        Reference image array against which changes are compared.
    users_vst : list
        List of users involved in changes for the current time step.
    users_sw_unique : array_like
        Array of unique users in the sliding window.
    t_inds_active_vst : list
        List of active time indices for validating changes against the stable version.
    agreeing_changes_vst : list
        List of changes that agree with the reference image over time.
    calc_attdef_userlist : bool
        Flag indicating whether to calculate lists of users associated with attack and defense changes.
    calc_inout_group : bool
        Flag indicating whether to calculate in-group and out-group changes and user metrics.
    calc_lifetime_vals : bool
        Flag indicating whether to calculate lifetime values for changes and users.
    save_ratio_pixels : bool
        Flag indicating whether to save the pixel-by-pixel attack/defense ratio arrays.
    save_ratio_images : str, optional
        Path to save the images of the pixel-by-pixel attack/defense ratio. If empty, images are not saved.

    Returns
    -------
    result : list
        A list containing unique users and, depending on flags, lists of users associated with specific types of changes.

    Notes
    -----
    The function updates the following attributes of the `cpst` object:
    - Number of changes (`n_changes`)
    - Number of defense changes (`n_defense_changes`)
    - Number of users (`n_users`)
    - Number of users participating in both attack and defense (`n_bothattdef_users`)
    - Number of users participating only in defense (`n_defenseonly_users`)
    - Number of users participating only in attack (`n_attackonly_users`)
    - Fraction of attack changes per image (`frac_attack_changes_image`)
    - n_users_new_vs_previoustime
    - n_users_sw
    - n_ingroup_changes
    - n_outgroup_changes
    - n_outgrouponly_users
    - n_ingrouponly_users
    - n_bothinout_users

    It also optionally calculates and saves images of the ratio of attack to defense changes for each pixel.
    """

    # get indices of all pixel changes that happen in the step (and that are part of the composition timerange for this pixel)
    cpst.n_changes.val[t_step] = len(t_inds_active)

    # Compare number of unique users to that in the previous timestep or sliding window
    users_now = cpart.user(t_inds_active)
    cpst.n_users.val[t_step] = len(np.unique(users_now))
    i_prev = (t_step-1) % cpst.sw_width
    cpst.n_users_new_vs_previoustime.val[t_step] = cpst.n_users.val[t_step] - len(np.intersect1d(users_now, users_vst[i_prev])) # count unique elements in common in the two arrays
    cpst.n_users_new_vs_sw.val[t_step] = cpst.n_users.val[t_step] - len(np.intersect1d(users_now, users_sw_unique))

    # Number of unique users in a sliding window
    users_vst[i_replace] = np.copy(users_now)
    users_sw_unique = np.unique(np.hstack(users_vst))
    cpst.n_users_sw.val[t_step] = len(users_sw_unique)

    # pixel changes that agree or not with the ref image
    agreeing_changes = np.array(ref_image[cpart.coordidx(t_inds_active)] == cpart.color(t_inds_active), np.bool_)
    disagree_changes = np.invert(agreeing_changes)
    cpst.n_defense_changes.val[t_step] = np.count_nonzero(agreeing_changes)

    # count users making defense or attack moves
    defense_users = np.unique(cpart.user(t_inds_active)[agreeing_changes])
    attack_users = np.unique(cpart.user(t_inds_active)[disagree_changes])
    attackdefense_users = np.intersect1d(attack_users, defense_users)
    cpst.n_bothattdef_users.val[t_step] = len(attackdefense_users)
    num_defense_users = len(defense_users)
    cpst.n_attackonly_users.val[t_step] = len(attack_users) - cpst.n_bothattdef_users.val[t_step]
    cpst.n_defenseonly_users.val[t_step] = num_defense_users - cpst.n_bothattdef_users.val[t_step]

    # Find pixel changes and users that belong to ingroup vs outgroup
    # to define in/out group, we compare the changes in the current time interval 
    # to both the forward and backward sliding window, which allows us to count innovations
    # that are eventually adopted into the composition as in-group changes. 
    #
    # For the forward comparison, we use the reference sliding window from the current step, 
    # then subtract the sliding window index with from the current time index, to reach the index before
    # the current sliding window. From this back-index, the current reference image is the forward sliding reference image. 
    # 
    # Doing this back referencing does require us to keep a record of the 
    # active indices and agreeing changes versus time, so we can always perform the back index to the proper time interval
    # to make the current sliding window stable reference the forward looking reference. 
    # TODO: if memory is a problem, we could delete the elements of agreeing_changes_vst and t_inds_active_vst
    # that come before the index [t_step - cpst.sw_width]. However, this would make the indexing less intuitive
    # so I am leaving it this way for now. 

    if calc_inout_group:
        # Save agreeing changes in this step for later comparison to...
        agreeing_changes_vst.append(agreeing_changes) 

        # Check if the current time step is beyond the sliding window width
        if t_step >= cpst.sw_width:
            # Determine the active indices 'forward' in time, accounting for the sliding window
            t_inds_active_fwd = t_inds_active_vst[t_step - cpst.sw_width]
            agreeing_changes_fwd = np.array(ref_image[cpart.coordidx(t_inds_active_fwd)] == cpart.color(t_inds_active_fwd), np.bool_)
            if t_step - cpst.sw_width >= cpst.sw_width_edge:
                agreeing_changes_bkwd = agreeing_changes_vst[t_step - cpst.sw_width]
                # ingroup changes are those that are either agreeing changes compared to the forward or backward looking sliding window reference
                ingroup_changes = agreeing_changes_bkwd | agreeing_changes_fwd  
            else:
                ingroup_changes = agreeing_changes_fwd

            outgroup_changes = np.invert(ingroup_changes)
            outgroup_inds = t_inds_active_fwd[outgroup_changes]
            idx = t_step + 1 - cpst.sw_width
            cpst.n_ingroup_changes.val[idx] = np.count_nonzero(ingroup_changes)
            cpst.n_outgroup_changes.val[idx] = np.count_nonzero(outgroup_changes)

            # users
            ingroup_users = np.unique(cpart.user(t_inds_active_fwd)[ingroup_changes])
            outgroup_users = np.unique(cpart.user(t_inds_active_fwd)[outgroup_changes])
            inout_users = np.intersect1d(ingroup_users, outgroup_users)
            cpst.n_bothinout_users.val[idx] = len(inout_users)
            num_ingroup_users = len(ingroup_users)
            cpst.n_outgrouponly_users.val[idx] = len(outgroup_users) - cpst.n_bothinout_users.val[idx]
            cpst.n_ingrouponly_users.val[idx] = num_ingroup_users - cpst.n_bothinout_users.val[idx]
            if cpst.ingroup_users_vst is not None:
                cpst.ingroup_users_vst[idx] = np.copy(ingroup_users)
                cpst.outgroup_users_vst[idx] = np.copy(outgroup_users)

    # count attack and defense changes for each pixel of the canvas
    if save_ratio_pixels:
        pixch_2Dcoor_offset = cpart.pixchanges_coords_offset(t_inds_active)  # for creation of the 2D images
        att = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)
        defe = np.zeros((cpart.width(1), cpart.width(0)), dtype=np.float16)

        for t in range(0, len(t_inds_active)):
            if agreeing_changes[t]:
                defe[pixch_2Dcoor_offset[1, t], pixch_2Dcoor_offset[0, t]] += 1
            else:
                att[pixch_2Dcoor_offset[1, t], pixch_2Dcoor_offset[0, t]] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            cpst.frac_attack_changes_image[t_step] = att / (defe + att)
        inds_nan = np.where((att>0) & (defe==0))
        cpst.frac_attack_changes_image[t_step][inds_nan[0], inds_nan[1]] = 100.

        if save_ratio_images != '':
            plt.figure()
            pcm = plt.pcolormesh(np.arange(0,cpart.width()[0]), np.arange(cpart.width()[1] - 1, -1, -1), cpst.frac_attack_changes_image[t_step], shading='nearest')
            plt.xlabel('x_pixel')
            plt.ylabel('y_pixel')
            plt.colorbar(pcm, label='# attack / # defense changes')
            plt.clim(0.99, 1.01)
            plt.savefig(os.path.join(var.FIGS_PATH, save_ratio_images, 'attack_defense_ratio_perpixel_'+time_str))
            plt.close()

    result = [users_sw_unique]
    if calc_attdef_userlist and calc_lifetime_vals:
        result.append(attack_users)
        result.append(defense_users)
        result.append(attackdefense_users)
    if calc_inout_group and t_step >= cpst.sw_width and calc_lifetime_vals:
        result.append(ingroup_users)
        result.append(outgroup_users)
        result.append(inout_users)
        result.append(outgroup_inds)

    return result


def create_files_and_get_sizes(t_lims, t_step, pixels,
                               file_size_png, file_size_bmp,
                               out_path_time, delete_bmp=True, delete_png=False):
    '''
    Creates bmp and png images from the 2D [pixels].
    Stores them, and records their sizes into file_size_png and file_size_bmp
    '''
    namecore = 'canvaspart_time{:06d}'.format(int(t_lims[t_step]))
    _, impath_png, impath_bmp = util.pixels_to_image(pixels, out_path_time, namecore + '.png', namecore + '.bmp')

    file_size_png[t_step] = util.get_file_size(impath_png)
    file_size_bmp[t_step] = util.get_file_size(impath_bmp)
    if delete_bmp:
        os.remove(impath_bmp)
    if delete_png:
        os.remove(impath_png)

def returnrate(current_color, prev_color, ref_color, inds_coor_active, last_time_installed_sw, t_lims, tstep):
    inds_att_prev = np.intersect1d(inds_coor_active, np.where(prev_color != ref_color)[0])
    n_att_prev = len(inds_att_prev)

    if n_att_prev == 0:
        return 1
    else:
        inds_not_recovered = inds_att_prev[ np.where(current_color[inds_att_prev] != ref_color[inds_att_prev])[0] ]
        inds_recoveredthenlost = np.count_nonzero(last_time_installed_sw[ref_color[inds_not_recovered], inds_not_recovered] > t_lims[tstep-1]) # will also necessarily be < t_lims[tstep]
        return (n_att_prev - len(inds_not_recovered) + inds_recoveredthenlost) / n_att_prev



def calc_changes_clustering(cpst, i, pixch_coor, pixel_coor, n_maxpix=100, n_MC=100, n_MC_trials=10):
    """
    Calculate Ripley's K function and average distance between pixel changes.
    """

    n_dist = len(cpst.ripley_distances)

    if pixch_coor.shape[1] < 2:
        for l in range(0, n_dist):
            cpst.ripley[l].val[i] = -1
        cpst.dist_average.val[i] = -1
        return    

    # at time step 0 or when the area changed, add a random set of pixel changes positions and calculate these values (MC). 
    # Take an average of multiple trials/experiments
    if (cpst.ripley_ref is None) or (cpst.avdist_ref is None):
        ripley_ref_alltrials = np.zeros((n_MC_trials, n_dist))
        avdist_ref_alltrials = np.zeros(n_MC_trials)
        for t in range(0, n_MC_trials): 
            mc_pixels = pixel_coor[:, np.random.choice(pixel_coor.shape[1], size=n_MC, replace=True)]
            distances = pdist(np.transpose(mc_pixels)) # distances between all pixel changes pairs

            for l,d in enumerate(cpst.ripley_distances):
                count = np.sum(distances <= d)
                ripley_ref_alltrials[t, l] = count * cpst.area_vst.val[i] / len(distances)
            avdist_ref_alltrials[t] = np.mean(distances)

        # average over trials
        cpst.ripley_ref = np.mean(ripley_ref_alltrials, axis=0)
        cpst.avdist_ref = np.mean(avdist_ref_alltrials)

    # Actual (non reference) calculation starts here
    # Randomly down sample the coordinates to speed up computation
    if pixch_coor.shape[1] > n_maxpix:
        inds_rdm = np.random.choice(np.arange(pixch_coor.shape[1]), size=n_maxpix, replace=False)
        pixch_coor = pixch_coor[:, inds_rdm]

    # Compute all pairwise distances
    distances = pdist(np.transpose(pixch_coor))  # returns condensed distance matrix

    # Ripley's K
    for l,d in enumerate(cpst.ripley_distances):
        count = np.sum(distances <= d)
        cpst.ripley[l].val[i] = count * cpst.area_vst.val[i] / len(distances) 

    # average distance between 2 pixel changes
    cpst.dist_average.val[i] = np.mean(distances)

    # normalized by reference values
    for l in range(0, n_dist):
        cpst.ripley_norm[l].val[i] = cpst.ripley[l].val[i] / cpst.ripley_ref[l]
    cpst.dist_average_norm.val[i] = cpst.dist_average.val[i] / cpst.avdist_ref
    return

def cross_correlation(cpst, i, itmin, r, binning, crosscorr_radial_ref):
    """
    Cross-correlation between images at t and t-1, summed for each separate color, then summed for each shift value.
    Use fftconvolve rather than correlate2d, to speed up the computation.
    """
    im1 = cpst.true_image[i]
    im2 = cpst.true_image[i-1 if i >= itmin else i]
    crosscorr = None
    # compute colors present in the images
    cols = np.flatnonzero(np.isin(np.arange(var.NUM_COLORS), im1) & np.isin(np.arange(var.NUM_COLORS), im2))

    # Calculate the cross-correlation between two images, for each shift value
    for c in (cols if len(cols) > 0 else [31]):
        mask1 = (im1 == c).astype(np.float32) # float32 faster for fftconvolve
        mask2 = (im2 == c).astype(np.float32)
        # for this color, subtract the crosscorr of the image with itself, from the crosscorr between t and t-1
        # which is equivalent to the crosscorr with the difference between the two images
        crosscorr_thiscol = fftconvolve(mask1, (mask1 - mask2)[::-1, ::-1], mode='full') # must flip the second image to get the cross-correlation
        if crosscorr is None:
            crosscorr = crosscorr_thiscol           
        else:
            crosscorr += crosscorr_thiscol

    # Calculate the number of non-zero crosscorr values at a certain radial shift
    crosscorr_radial, _ = np.histogram(r, bins=binning, 
                                       weights=crosscorr.ravel().astype(int)) # sum all the crosscorr values at some r value
    
    for l in range(len(cpst.crosscorr_distances)):
        cpst.crosscorr[l].val[i] = crosscorr_radial[l] / crosscorr_radial_ref[l] / cpst.area_vst.val[i] # area not strictly necessary, but normalizes crosscorr for radius=0
        cpst.crosscorr[l].val[i] /= 1 if (l==0 or cpst.crosscorr[0].val[i]==0) else cpst.crosscorr[0].val[i]
    return

def returntime(last_time_installed_sw, last_time_removed_sw,
               current_color, ref_color,
               inds_coor_active, t, sw_width_sec):
    '''
    Calculate return time, as the time each pixel spent in the non-reference color during the last attack within the SW
    consider only pixels that had that form during the SW: ...-D-A-D or ...-D-A,
    Exclude D only, and A-D (with A starting before start of SW).
    '''
    indsall = np.arange(last_time_removed_sw.shape[1])
    last_time_removed_refcol = last_time_removed_sw[ref_color[indsall], indsall]
    # indices where pixels are in defense (ref) color, and last attack happened within the sliding window
    inds_def_attackinsw = np.intersect1d(inds_coor_active,
                                         np.where((current_color == ref_color) & (last_time_removed_refcol > t - sw_width_sec))[0])
    returntime_indefpix = last_time_installed_sw[ref_color[inds_def_attackinsw], inds_def_attackinsw] - last_time_removed_refcol[inds_def_attackinsw]

    # indices where pixels are in an attack color
    inds_att = np.intersect1d(inds_coor_active, np.where(current_color != ref_color)[0])
    returntime_inattpix = t - last_time_removed_refcol[inds_att]

    # compute mean and quantiles of returntime over pixels
    returntime_all = np.concatenate((returntime_inattpix, returntime_indefpix))
    return util.pixel_integrated_stats(returntime_all, 0)


def return_time_fwdlooking(cpart, ref_image, t_lims=[0, var.TIME_TOTAL], summary_stats=True):
    '''
    Return time for all attack pixel changes: the time that each pixel
    that was changed from its reference color (in ref_image) to return
    to the reference color. Also returns the time at which those pixels wer attacked.
    '''

    # Canvas part data
    sortcoor = cpart.pixch_sortcoord  # sort pixel changes by pixel coordinate
    color = cpart.pixel_changes['color'][sortcoor]
    coordidx = cpart.pixel_changes['coord_index'][sortcoor]
    time = cpart.pixel_changes['seconds'][sortcoor]

    pixch_2Dcoor_offset = cpart.pixchanges_coords_offset(sortcoor)
    xcoords = pixch_2Dcoor_offset[0]
    ycoords = pixch_2Dcoor_offset[1]

    # should be of form [pixel 1 att(False), def(True), att, att, ..., pixel 2 def,def,att,def,att,...]
    agreeing_changes = np.array( ref_image[ycoords, xcoords] == color, np.bool_)
    disagree_changes = np.invert(agreeing_changes)
    # does the pixel change correspond to a new pixel?
    coord_changed = np.hstack(( True, (np.diff(coordidx) > 0) ))
    coord_change_ind = np.where(coord_changed)[0]
    # is this pixel change the beginning of an attack or defense sequence?
    newly_attacked_or_restored = np.hstack((True, np.diff(agreeing_changes)))
    newly_attacked_or_restored[coord_changed & disagree_changes] = True # consider that it is newly attacked when looking at another pixel
    newly_attacked_or_restored[coord_changed & agreeing_changes] = False # defense changes at the start of a new pixel are ignored

    beg_att_seq_ind = np.where(disagree_changes & newly_attacked_or_restored)[0]
    beg_def_seq_ind = np.where(agreeing_changes & newly_attacked_or_restored)[0]
    for i in range(0, len(coord_change_ind) - 1):
        thispix = np.arange(coord_change_ind[i], coord_change_ind[i+1])
        new_att = np.intersect1d(beg_att_seq_ind, thispix, assume_unique=True)
        new_def = np.intersect1d(beg_def_seq_ind, thispix, assume_unique=True)
        if len(new_att) == len(new_def) + 1:
            unrestored_att_ind = new_att[-1]
            new_att = new_att[:-1]
            returntime += [var.TIME_TOTAL]
            time_attackflip += [time[unrestored_att_ind]]
        returntime += list(time[new_def] - time[new_att])
        time_attackflip += list(time[new_att])
    returntime = np.array(returntime)
    time_attackflip = np.array(time_attackflip)

    returntime_tbinned = None
    returntime_mean = None
    returntime_median_overln2 = None
    if summary_stats:
        returntime_tbinned = np.empty(len(t_lims)-1, dtype=object)
        returntime_mean = np.empty(len(t_lims)-1)
        returntime_median_overln2 = np.empty(len(t_lims)-1)
        timebin_ind = np.digitize(time_attackflip, t_lims) - 2  # TODO: check this
        for j in range(0, len(t_lims)-1):  # initialize with empty lists
            returntime_tbinned[j] = []
        for j in range(0, len(returntime)):  # fill the histogram-like array using the result from np.digitize
            returntime_tbinned[timebin_ind[j]].append(returntime[j])
        for j in range(0, len(t_lims)-1):  # cannot use more clever numpy because the lists in axis #3 are of different sizes
            returntime_mean[j] = np.mean(np.array(returntime_tbinned[j]))
            returntime_median_overln2[j] = np.median(np.array(returntime_tbinned[j]))
        returntime_median_overln2 /= np.log(2)

    return (returntime, time_attackflip, returntime_tbinned, returntime_mean, returntime_median_overln2)
