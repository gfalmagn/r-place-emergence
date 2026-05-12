import numpy as np
import os
import pandas as pd
import rplacem.compute_variables as comp
import rplacem.utilities as util
import rplacem.transitions as tran
import rplacem.time_series as ts
from rplacem import var as var
import shutil
import warnings
import math



class CanvasPartStatistics(object):
    ''' Object containing all the important time-dependent variables concerning the input CanvasPart

    attributes
    ----------
    GENERAL
    id : str
        Corresponds to the out_name() of the CanvasPart.
        Is unique for compositions (where it is = cpart.id) and for rectangles (cpart.is_rectangle == True)
        Can designate different canvas parts in other cases.
            Set in __init__()
    info : AtlasInfo
        Direct AtlasInfo object from the CanvasPart
    quadrant : int
        the index of the earliest canvas quadrant that the composition was on (from 0 to 2 for 2022, from 0 to 6 for 2023)
    compute_vars: dictionary (string to int)
        says what level of information to compute and store for each variable.
        The keys for the computed variables are:
            'stability', 'other', 'entropy', 'transitions', 'attackdefense'
        The levels mean:
            0: not computed
            1: basic variables are computed
            2: images are created and stored in the form of 2d numpy arrays of pixels containing color indices.
                Variables based on these images are also computed
            3: these pixel arrays are transformed to images and stored. Also makes a movie from stored images.
            4: more extensive info is stored in the class and in a pickle file.
        Set in __init__()
    verbose : boolean
        Saying if output is printed during running member functions
            Set in __init__()

    AREA
    area : int
        number of pixels (any pixels that are active at some time point) of the canvas part
            Set in __init__()
    area_rectangle : int
        size (in pixels) of the smallest rectangle encompassing the whole composition.
        Equals area when the composition is such a rectangle at some time point.
            Set in __init__()
    area_vst : TimeSeries, n_pts = n_t_bins+1
        Number of active pixels at each time step.
            Needs compute_vars['attackdefense'] > 0
            Set in comp.n_changes_and_users(), in count_attack_defense_events().
    stable_borders_timeranges : 2d numpy array, shape (# of stable-border timeranges, 2)
        continuous time ranges over which the border_path of the CanvasPart does not change significantly

    TIME BINS
    n_t_bins: int
        number of time intervals
            Set in __init__()
    t_interval: float
        width of time interval
            Set in __init__()
    t_lims : 1d array of size n_t_bins+1
        limits of the time bins in which all variables are computed
           Set in __init__()
    t_unit : float
        time unit used for normalizing some variables.
            Set in __init__()
    t_norm : float
        time normalisation (t_interval / t_unit).
            Set in __init__()
    tmin, tmax: floats
        time range of the composition, on which to compute the variables.
            Set in __init__()
    sw_width_sec: float
        Size [in seconds] of the sliding window over which the sliding reference image is computed
    sw_width: int
        Size [in units of t_interval] of the sliding window over which the sliding reference image is computed

    TRANSITIONS
    transition_param : list of floats
        cutoff parameters for the tran.find_transitions() function (see doc of this function for meaning of params)
        The last two arguments are in seconds

    VARIABLES
    STABILITY
    stability: TimeSeries, n_pts = n_t_bins+1
        stability averaged over all pixels of the CanvasPart, for each time bin.
            Needs compute_vars['stability'] > 0
            Set in comp.main_variables()
    runnerup_timeratio: array of 4 TimeSeries, n_pts = n_t_bins+1
            For each time step, contains 4 pixel-integrated stats (mean, median, and 90th percentile, mean of highest decile) of runnerup_time
            It is the ratio of time spent in the second most used color to the time in the most stable color
            Needs compute_vars['stability'] > 0
            Set in comp.main_variables()
    n_used_colors:  array of 4 TimeSeries, n_pts = n_t_bins+1
            For each time step, contains 4 pixel-integrated stats (mean, median, and 90th percentile, mean of highest decile) of n_used_colors
            It is the number of used colors in a pixel in a time interval
            Needs compute_vars['stability'] > 0
            Set in comp.main_variables()
    instability_norm : TimeSeries, n_pts = n_t_bins+1
        Time-normalised instability: (1 - stability_vst) / t_norm
            Needs compute_vars['stability'] > 0
            Set in __init__()
    stable_image
    second_stable_image
    third_stable_image : 1d array of pixels images (2d numpy array containing color indices)
        Most stable image (and second and third most stable images) in each time bin
            Needs compute_vars['stability'] > 1
            Set in comp.main_variables()
    size_compr_stab_im : TimeSeries, n_pts = n_t_bins+1
        size of the compressed stable image
    entropy_stab_im : TimeSeries, n_pts = n_t_bins+1
        Ratio of size of the size of the compressed stable image pixel array to the number of active pixels, at each time step
        Set in ratios_and_normalizations()

    CLASSIC EWS -- need compute_vars['ews'] > 0
    autocorrelation: TimeSeries, npts = n_t_bins + 1
        calculated based on 4 cases of current color being equal or not to the previous color and to the mode color.
    variance: TimeSeries, npts = n_t_bins + 1
        variance as calculated by the spread in time fractions of different colors. An alternative variance is
        1-stability.


    ENTROPY -- all need compute_vars['entropy'] > 0
    diff_pixels_stable_vs_swref
    diff_pixels_inst_vs_swref
    diff_pixels_inst_vs_swref_forwardlook
    diff_pixels_inst_vs_inst
    diff_pixels_inst_vs_stable : TimeSeries, n_pts = n_t_bins+1
        Number of pixels that differ between images:
        stable over the timestep, instantaneous at the end of this or the previous timestep ("inst"),
        or the reference image over the sliding window ("swref")
        or over a forward-looking future sliding window ("swref_forwardlook")
            Set in comp.main_variables()
    frac_pixdiff_stable_vs_swref
    frac_pixdiff_inst_vs_swref
    frac_pixdiff_inst_vs_swref_forwardlook
    frac_pixdiff_inst_vs_inst_norm
    frac_pixdiff_inst_vs_stable_norm : TimeSeries, n_pts = n_t_bins+1
        Same as above, but divided the number of active pixels.
        Also normalized by t_norm for the last two.
            Set in ratios_and_normalizations()
    size_uncompressed :
    size_compressed : TimeSeries, n_pts = n_t_bins+1
        Size of the uncompressed or compressed image (of the canvas part) file at the end of each time step
            Needs compute_vars['entropy'] > 0 (or > 3 for size_compressed)
            Set in comp.main_variables()
    entropy_bpmnorm : TimeSeries, n_pts = n_t_bins+1
        Ratio of sizes of the png and bmp image files at the end of each time step
            Set in ratios_and_normalizations()
    entropy : TimeSeries, n_pts = n_t_bins+1
        Ratio of size of the png image file to the number of active pixels, at each time step
            Set in ratios_and_normalizations()
    true_image : 1d array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        True image of the canvas part at the end of the time interval
            Needs compute_vars['entropy'] > 1
            Set in comp.main_variables()
    fractal_dim_mask_median : TimeSeries, n_pts = n_t_bins+1
        Median fractal dimension across all colors, ignoring zeros
    fractal_dim_weighted : TimeSeries, n_pts = n_t_bins+1
        Weighted fractal dimension across all colors

    TRANSITIONS -- all need compute_vars['transitions'] > 0
    n_transitions : int
        Number of found transitions
            Set in search_transitions()
    transition_tinds
    transition_times: 2d array, shape (number of transitions, 6)
        Delimiting times, and indices of t_lims, for each transition.
        For each transition, is of the form
        [beg, end of pre stable period, beg, end of transition period, beg, end of post stable region]
            Set in tran.find_transitions(), in search_transitions()
    trans_start_time
    trans_start_tind : array of size n_transitions
        Time (and indices in t_lims) at which most of the variables characterizing the transition have reached
        their half maximum, as computed in tran.transition_start_time()
            Set in search_transitions()
    refimage_pretrans
    refimage_intrans
    refimage_posttrans : array of size min(1, number of transitions), of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Reference stable image over the sliding window before the transition (sliding window ending at the end
        of the pre-transition stable period) or after the transition (sliding window starting at the beginning
        of the post-transition stable period).
        refimage_intrans is the true_image at time trans_start_time
            Needs compute_vars['transitions'] > 1
            Set in search_transitions()
    frac_diff_pixels_pre_vs_post_trans : 1d array of shape (n_transitions)
        Fraction of the active pixels that differ between the pre- and post-transition stable images
            Set in search_transitions()

    ATTACK-DEFENSE -- all need compute_vars['attackdefense'] > 0
    refimage_sw : 1d array of 2D pixels image (length n_t_bins+1)
        For each timestep of index i, it is the stable image over the interval [i - sw_width, i],
        used as reference image for all the attack/defense variables
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    refimage_sw_flat : 1D array of 1D pixels image (length n_t_bins+1)
        same than refimage_sw, but used only temporarily for pixel differences before and after transition.
        Is removed after being used.
    n_changes : TimeSeries, n_pts = n_t_bins+1
        Number of pixel changes in each time bin
            Set in comp.main_variables()
    n_changes_norm :
        same as n_changes, normalized by t_norm and area_vst
            Set in ratios_and_normalizations()
    frac_attack_changes : 2d array of floats, shape (# transitions, n_t_bins)
        Fraction of pixel changes that are attacking (compared to sliding window reference), in each time bin
            Set in ratios_and_normalizations()
    n_users_total : float
        total number of user having contributed to the composition between tmin and tmax
            Set in comp.main_variables()
    n_users : TimeSeries, n_pts = n_t_bins+1
        number of users that changed any pixels in this composition in each time range
            Set in comp.main_variables()
    n_users_norm :
        same as n_users_vst, normalized by t_norm and area_vst
            Set in ratios_and_normalizations()
    n_users_sw : TimeSeries, n_pts = n_t_bins+1
        Number of unique users active within the preceding time window (including present timestep)
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    n_users_sw_norm : TimeSeries, n_pts = n_t_bins+1
        Same as above, normalized by t_unit and the active area
            Set in ratios_and_normalizations()
    frac_users_new_vs_previoustime
    frac_users_new_vs_sw : TimeSeries, n_pts = n_t_bins+1
        Fraction of unique users in this timestep that were also active in the last timestep,
        or in the preceding sliding window
            Set in ratios_and_normalizations()
    n_users_new_vs_previoustime
    n_users_new_vs_sw : TimeSeries, n_pts = n_t_bins+1
        Same as above, but without dividing by n_users_sw.
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    changes_per_user_sw : TimeSeries, n_pts = n_t_bins+1
        number of pixel changes per unique user active within the preceding
        sliding window (including the present timestep)
            Set in ratios_and_normalizations()
    frac_attackonly_users
    frac_defenseonly_users
    frac_bothattdef_users : TimeSeries, n_pts = n_t_bins+1
        fraction of n_users that contributed pixels that are consistent refimage_sw,
        or not (attackonly), or fraction of users that did both (bothattdef)
            Set in ratios_and_normalizations()
    n_defenseonly_users
    n_attack_users
    n_bothattdef_users :
        Same as above, but without dividing by n_users.
        Kept if compute_vars['attackdefense'] > 1
            Set in comp.main_variables()
    n_ingroup_changes
    n_outgroup_changes
    n_ingrouponly_users
    n_outgrouponly_users
    n_bothinout_users : TimeSeries, n_pts = n_t_bins+1
        In-group / out-group change and user counts per time bin.
            Needs compute_vars['inoutgroup'] > 0.
            Set in comp.main_variables() and comp.num_changes_and_users()
    ingroup_users_vst 
    outgroup_users_vst: 1d object array of length n_t_bins+1
        Unique user IDs with at least one in-group (or out-group) change in each time bin.
        Kept if compute_vars['inoutgroup'] > 1.
            Set in comp.main_variables() and comp.num_changes_and_users()
    frac_attack_changes_image : array of size n_t_bins, of 2d numpy arrays.
        Each element is a pixels image info (2d numpy array containing color indices)
        Image containing, for each pixel, the fraction of pixel changes that are attacking, for each time step
        Needs compute_vars['attackdefense'] > 1
    returntime : array of 4 TimeSeries
        For each time step, contains 4 pixel-integrated stats (mean, median, and 90th percentile, mean of highest decile) of return time
        which is the time that an attack within the sliding window takes to return to a defense pixel
            Set in comp.main_variables()
    cumul_attack_timefrac : TimeSeries, n_pts = n_t_bins+1
        Sum of the times that each pixel spent in an attack color during this timestep
        Normalized by the timestep width times #pixels
            Set in comp.main_variables()

    OTHER: -- all need compute_vars['other'] > 0
    n_moderator_changes
    n_cooldowncheat_changes
    n_redundant_color_changes
    n_redundant_coloranduser_changes
        number of pixel changes that:
        - were issued by moderators
        - break the 5-minute per-user cooldown
        - are redundant in color with what the pixel contained before the change
        - are redundant in color and user with what the pixel contained before the change
            Set in comp.main_variables()
    frac_moderator_changes
    frac_cooldowncheat_changes
    frac_redundant_color_changes
    frac_redundant_coloranduser_changes
        Same as above, but divided by the number of pixel changes in this timestep
            Set in ratios_and_normalizations()

    VOID ATTACKS -- need compute_vars['void_attack'] > 0
    frac_black_px: TimeSeries, npts = n_t_bins + 1
        fraction of black pixels in composition versus time
    frac_purple_px: TimeSeries, npts = n_t_bins + 1
        fraction of purple pixels in composition versus time
    frac_black_ref: TimeSeries, npts = n_t_bins + 1
        fraction of black pixels in reference image versus time
    frac_purple_ref: TimeSeries, npts = n_t_bins + 1
        fraction of purple pixels in reference image versus time

    CLUSTERING -- need compute_vars['clustering'] > 0
    ripley : array of [ripley_distances] TimeSeries, n_pts = n_t_bins+1
        Ripley's K function, for 10 different distance bins, recorded in cpst.ripley_distances
    crosscorr : array of [crosscorr_distances] TimeSeries, n_pts = n_t_bins+1
        Cross-correlation between images a t and t-1, for different radial shift values r, normalised by r^2

    methods
    -------
    private:
        __init__
    protected:
    public:
        ts_init()
        checks_warnings()
        keep_all_basics()
            Check if all basic variables are asked for (compute_vars[:] > 0)
        ratios_and_normalizations()
            Normalize variables
        returntime_stats()
            Calculate various variables for the full returntime array
        search_transitions()
            Look for transitions and set associated attributes
        fill_timeseries_info()
            User-called function that fills all the textual attributes of 1d time-series variables
    '''

    def __init__(self,
                 cpart,
                 t_interval=300,
                 tmax=var.TIME_TOTAL,
                 compute_vars={'stability': 3, 'entropy': 3, 'transitions': 3, 'attackdefense': 3, 'other': 1, 'ews': 0, 'void_attack': 0, 'inoutgroup': 0, 'lifetime_vars': 0, 'clustering': 1},
                 sliding_window=14400,
                 sliding_window_edge = 3600,
                 trans_param=[0.3, 6],#, 2*3600, 4*3600],
                 timeunit=300,  # 5 minutes
                 verbose=False,
                 renew=True,
                 dont_keep_dir=False,
                 flattening='hilbert_pkg',
                 compression='LZ77',
                 keep_ref_im_const=False
                 ):
        '''
        cpart : CanvasPart object
            That associated to this CanvasPartStatistics
        renew : boolean
            Delete the output directories if they exist, before re-running
        dont_keep_dir : boolean
            Remove directory after running, if it did not exist before
        compute_vars : string to int dictionary
            Gives level of information to compute for each variable. Check class doc for details.
        verbose: boolean
            Is output printed during running. Check class doc for details.
        trans_param : list of floats
            Fills transition_param attribute. Check class doc for details.
        tmax : float
            Fills tmax attribute. Check class doc for details.
        timeunit: float
            Fills t_unit attribute. Check class doc for details.
        n_tbins : int
            Fills n_t_bins attribute. Check class doc for details.
        sliding_window : float, in seconds
            Duration of the sliding window on which the reference image is computed
        sliding_window_edge : float, in seconds
            Duration of the minimum sliding window on which the reference image is computed, when at the end edge of the composition lifetime
        t_interval : float, in seconds
            width of the time bins
        keep_ref_im_const : boolean
            If True, then uses the most stable image over the specified time range as the reference image, rather than a sliding window reference image.
        '''

        self.info = cpart.info
        self.id = cpart.out_name()
        if compute_vars['stability'] < 3 and compute_vars['entropy'] < 3:
            renew = False
        dirpath = os.path.join(var.FIGS_PATH, str(cpart.out_name()))
        dir_exists = util.make_dir(dirpath, renew)

        self.compute_vars = compute_vars
        self.verbose = verbose
        self.area = cpart.num_pix()
        self.area_rectangle = cpart.width(0) * cpart.width(1)
        self.tmin, self.tmax = cpart.min_max_time(tmax_global=tmax)
        self.quadrant = cpart.tmin_quadrant()[1]
        if self.tmin > 1e-4:
            start_pixels = cpart.start_pixels()
        else:
            start_pixels = None

        self.t_interval = t_interval # seconds
        self.n_t_bins = math.ceil((self.tmax - self.tmin) / self.t_interval)
        self.t_lims = self.calc_t_lims()
        self.t_unit = timeunit
        self.t_norm = self.t_interval / self.t_unit
        self.sw_width = int(sliding_window/self.t_interval)
        self.sw_width_sec = self.sw_width * self.t_interval # force the sliding window to be a multiple of the time interval
        self.sw_width_edge = int(sliding_window_edge/self.t_interval)

        # Creating attributes that will be computed in comp.main_variables()
        self.diff_pixels_stable_vs_swref = ts.TimeSeries()
        self.diff_pixels_inst_vs_swref = ts.TimeSeries()
        self.diff_pixels_inst_vs_swref_forwardlook = ts.TimeSeries()
        self.diff_pixels_inst_vs_inst = ts.TimeSeries()
        self.diff_pixels_inst_vs_stable = ts.TimeSeries()
        self.area_vst = ts.TimeSeries()

        self.stability = None
        self.runnerup_timeratio = None
        self.n_used_colors = None
        self.size_compr_stab_im = ts.TimeSeries()
        self.stable_image = None
        self.second_stable_image = None
        self.third_stable_image = None
        self.refimage_sw = None
        self.true_image = None
        self.attack_defense_image = None
        self.frac_attack_changes_image = None

        self.autocorr_bycase = ts.TimeSeries()
        self.autocorr_bycase_norm = ts.TimeSeries()
        self.autocorr_multinom = ts.TimeSeries()
        self.autocorr_subdom = ts.TimeSeries()
        self.autocorr_dissimil = ts.TimeSeries()
        self.variance_subdom = ts.TimeSeries()
        self.variance_multinom = ts.TimeSeries()
        self.variance2 = ts.TimeSeries()
        self.variance_from_frac_pixdiff_inst = ts.TimeSeries()

        self.n_changes = ts.TimeSeries()
        self.n_defense_changes = ts.TimeSeries()
        self.n_users = ts.TimeSeries()
        self.n_users_total = None
        self.n_bothattdef_users = ts.TimeSeries()
        self.n_defenseonly_users = ts.TimeSeries()
        self.n_attack_users = ts.TimeSeries()
        self.n_bothattdef_users_lifetime = None
        self.n_defenseonly_users_lifetime = None
        self.n_attackonly_users_lifetime = None

        self.n_ingroup_changes = ts.TimeSeries()
        self.n_outgroup_changes = ts.TimeSeries()
        self.n_ingrouponly_users = ts.TimeSeries()
        self.n_outgrouponly_users = ts.TimeSeries()
        self.n_bothinout_users = ts.TimeSeries()
        self.ingroup_users_vst = None
        self.outgroup_users_vst = None
        self.num_edge_pixels = ts.TimeSeries()
        self.n_bothinout_users_lifetime = None
        self.n_ingrouponly_users_lifetime = None
        self.n_outgrouponly_users_lifetime = None
        self.outgroup_inds = None

        self.returntime = None
        self.returnrate = ts.TimeSeries()
        self.cumul_attack_timefrac = ts.TimeSeries()
        self.frac_black_px = ts.TimeSeries()
        self.frac_purple_px = ts.TimeSeries()
        self.frac_black_ref = ts.TimeSeries()
        self.frac_purple_ref = ts.TimeSeries()

        self.n_moderator_changes = ts.TimeSeries()
        self.n_cooldowncheat_changes = ts.TimeSeries()
        self.n_redundant_color_changes = ts.TimeSeries()
        self.n_redundant_coloranduser_changes = ts.TimeSeries()

        self.n_users_new_vs_previoustime = ts.TimeSeries()
        self.n_users_new_vs_sw = ts.TimeSeries()
        self.n_users_sw = ts.TimeSeries()

        self.size_compressed = ts.TimeSeries()
        self.size_uncompressed = ts.TimeSeries()

        self.fractal_dim_mask_median = ts.TimeSeries()
        self.fractal_dim_weighted = ts.TimeSeries()
        self.complexity_multiscale = ts.TimeSeries()
        self.complexity_levenshtein = ts.TimeSeries()
        self.wavelet_high_freq = ts.TimeSeries()
        self.wavelet_mid_freq = ts.TimeSeries()
        self.wavelet_low_freq = ts.TimeSeries()
        self.wavelet_low_freq_tm = ts.TimeSeries()
        self.wavelet_high_freq_tm = ts.TimeSeries()
        self.ssim_stab_ref = ts.TimeSeries()

        self.ripley = None # array of TimeSeries, created in compute_variables
        typicalsize = math.sqrt(self.area_rectangle)
        self.ripley_distances = np.array([2, max(3, int(0.2*typicalsize)), max(5, int(0.5*typicalsize))]) #np.array([1, 2, 3, 4, 6, 8, 10, 15, 23, 35, 50])
        self.ripley_norm = None 
        self.dist_average = ts.TimeSeries()
        self.dist_average_norm = ts.TimeSeries()
        self.crosscorr = None # array of TimeSeries, created in compute_variables
        dist_tmp = [0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30, 35, 40, 50, 70, 100, 150]
        self.crosscorr_distances = []
        for d in dist_tmp:
            if d <= int(0.8 * typicalsize):
                self.crosscorr_distances.append(d)
        self.crosscorr_distances = np.array(self.crosscorr_distances)
        self.image_shift_min = ts.TimeSeries()
        self.image_shift_minpos = ts.TimeSeries()
        self.image_shift_slope = ts.TimeSeries()

        self.transition_param = trans_param
        self.n_transitions = None

        self.checks_warnings()

        if keep_ref_im_const:
            ref_im_const = self.calc_ref_image_tot(cpart)
        else:
            ref_im_const = None

        # find continuous time ranges over which the border_path of the composition does not change significantly
        self.stable_borders_timeranges = cpart.stable_borderpath_timeranges()

        # Magic happens here
        comp.main_variables(cpart, self, print_progress=self.verbose, delete_dir=dont_keep_dir,
                            flattening=flattening,
                            compression=compression,
                            start_pixels=start_pixels,
                            ref_im_const=ref_im_const)

        # extract shift quantities from cross correlation: minimum, position of the minimum, and slope of a linear fit
        if self.compute_vars['clustering'] > 0:
            crosscorr_all = np.array([self.crosscorr[i].val for i in range(1, len(self.crosscorr_distances))]).T
            self.image_shift_minpos.val = self.crosscorr_distances[1+np.argmin(crosscorr_all, axis=1)]
            self.image_shift_min.val = np.min(crosscorr_all, axis=1)
            # slope using the direct least squares formula
            w = np.asarray([1]*(int(len(self.crosscorr_distances)/2)-1) # weights to increase the influence of sparser high distance points
                        + [2]*(len(self.crosscorr_distances)-int(len(self.crosscorr_distances)/2)))
            w_sum = np.sum(w)
            x_mean = np.sum(w * self.crosscorr_distances[1:]) / w_sum
            y_mean = np.sum(w * crosscorr_all, axis=1, keepdims=True) / w_sum
            x_c = self.crosscorr_distances[1:] - x_mean
            y_c = crosscorr_all - y_mean
            self.image_shift_slope.val = np.sum(w * x_c * y_c, axis=1) / np.sum(w * x_c ** 2)
            del crosscorr_all, x_c, y_c, y_mean, x_mean, w, w_sum
            if self.compute_vars['clustering'] <= 2:
                del self.crosscorr, self.ripley, self.dist_average # keep only ripley_norm and dist_average_norm

        # ratio variables and normalizations
        self.ratios_and_normalizations()

        if np.any(self.frac_users_new_vs_sw.val > 1):
            print('frac_users_new_vs_sw > 1 !!!!!!!')
        if np.any(self.variance2.val < 1):
            print('variance2 < 1 !!!!!!!')

        # Make movies
        if self.compute_vars['stability'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTimeStab'), fps=15)
        if self.compute_vars['entropy'] > 2:
            util.save_movie(os.path.join(dirpath, 'VsTimeInst'), fps=15)
        if self.compute_vars['attackdefense'] > 2:
            util.save_movie(os.path.join(dirpath, 'attack_defense_ratio'), fps=15)
            util.save_movie(os.path.join(dirpath, 'attack_defense_Ising'), fps=6)

        # find transitions
        if compute_vars['transitions'] > 0:
            self.search_transitions(cpart)
            if compute_vars['transitions'] < 2:
                self.refimage_pretrans = None
                self.refimage_posttrans = None
        else:
            self.n_transitions = 0

        if compute_vars['transitions'] > 1 or compute_vars['entropy'] > 0:
            # calculate variance over ~10 most recent timesteps
            def rolling_mean_squares(v, nroll):
                sq = pd.Series(0.5 * (nroll/(nroll-1)) * v**2)
                return np.array(sq.rolling(window=nroll, min_periods=1).mean())
            self.variance_from_frac_pixdiff_inst.val = rolling_mean_squares(self.frac_pixdiff_inst_vs_inst_norm.val, 
                                                                            self.frac_pixdiff_inst_vs_inst_norm.sw_width_ews)

        # calculate kendall tau's
        self.returnrate.set_kendall_tau()
        self.returntime[0].set_kendall_tau()
        if self.compute_vars['stability'] > 0:
            self.instability_norm[0].set_kendall_tau()
        self.variance_multinom.set_kendall_tau()
        self.variance_subdom.set_kendall_tau()
        self.variance2.set_kendall_tau()
        self.autocorr_bycase.set_kendall_tau()
        self.autocorr_bycase_norm.set_kendall_tau()
        self.autocorr_multinom.set_kendall_tau()
        self.autocorr_subdom.set_kendall_tau()
        self.autocorr_dissimil.set_kendall_tau()

        # Memory savings here 
        if compute_vars['attackdefense'] < 2:
            self.n_defense_changes = ts.TimeSeries()
            self.n_defenseonly_users = ts.TimeSeries()
            self.n_attack_users = ts.TimeSeries()
            self.n_bothattdef_users = ts.TimeSeries()
            self.n_users_sw = ts.TimeSeries()

            self.n_users_new_vs_previoustime = ts.TimeSeries()
            self.n_users_new_vs_sw = ts.TimeSeries()
            self.refimage_sw = None
        self.refimage_sw_flat = None
        if compute_vars['entropy'] < 2:
            self.size_compressed = ts.TimeSeries()
            self.diff_pixels_stable_vs_swref = ts.TimeSeries()
            self.diff_pixels_inst_vs_swref_forwardlook = ts.TimeSeries()
            self.diff_pixels_inst_vs_inst = ts.TimeSeries()
            self.diff_pixels_inst_vs_stable = ts.TimeSeries()
            self.true_image = None

        # remove directory if it did not exist before and if dont_keep_dir
        if (not dir_exists) and dont_keep_dir:
            shutil.rmtree(dirpath)

    def calc_t_lims(self, eps=0.01):
        '''
        Calculate the t_lims. The last t_lim is t_max, even
        if that means altering the size of the last interval between times
        '''

        # Subtract eps to make sure the final value stays under tmax + t_interval
        t_lims = np.arange(self.tmin, self.tmax + self.t_interval - eps, self.t_interval)

        # Final value should be tmax, even if it means the last time interval is not equal to the rest
        t_lims[-1] = self.tmax
        return t_lims

    def calc_ref_image_tot(self, cpart):
        t_lims = self.t_lims
        t_inds = cpart.intimerange_pixchanges_inds(t_lims[0], t_lims[-1])
        seconds = cpart.pixel_changes['seconds']
        color = cpart.pixel_changes['color']
        pixch_coord_inds = cpart.pixel_changes['coord_index']
        current_color = cpart.start_pixels()
        last_time_installed_sw = comp.initialize_start_time_grid(cpart, t_lims[0], add_color_dim=True)
        last_time_removed_sw = np.copy(last_time_installed_sw)

        time_spent_in_color_tot_time = comp.calc_time_spent_in_color(cpart,
                                                                     seconds[t_inds],
                                                                     color[t_inds],
                                                                     pixch_coord_inds[t_inds],
                                                                     t_lims[0],
                                                                     t_lims[-1],
                                                                     current_color,
                                                                     last_time_installed_sw,
                                                                     last_time_removed_sw)

        ref_im_tot = comp.calc_stable_cols(time_spent_in_color_tot_time)[:, 0]

        return ref_im_tot

    def checks_warnings(self):
        if np.any(np.diff(self.t_lims) < 0):
            warnings.warn('The array of time limits t_lims should contain increasing values. It will be modified to fit this requirement.')
            for i in range(1, self.n_t_bins + 1): # need for loop because it needs to be done in that order
                if self.t_lims[i] < self.t_lims[i-1]:
                    self.t_lims[i] = self.t_lims[i-1]
        if self.sw_width == 0:
            warnings.warn('The interval size is larger than the sliding window. Choose a smaller interval size or a larger sliding window.')

    def ts_init(self, val):
        return ts.TimeSeries(val=val, cpstat=self)

    def keep_all_basics(self):
        return self.compute_vars['entropy'] > 0 \
           and self.compute_vars['stability'] > 0 \
           and self.compute_vars['attackdefense']> 0 \
           and self.compute_vars['transitions'] > 0 \
           and self.compute_vars['other'] > 0

    def ratios_and_normalizations(self):
        if self.compute_vars['stability'] > 0:
            self.instability_norm = np.empty(4, dtype=object)
            for k in range(0, 4):
                self.instability_norm[k] = self.ts_init((1 - self.stability[k].val) / self.t_norm)

        self.n_changes_norm = self.ts_init(util.divide_treatzero(self.n_changes.val / self.t_norm, self.area_vst.val, 0, 0))
        if self.compute_vars['attackdefense'] > 0:
            self.n_users_norm = self.ts_init(util.divide_treatzero(self.n_users.val / self.t_norm, self.area_vst.val, 0, 0))
            self.n_users_sw_norm = self.ts_init(util.divide_treatzero(self.n_users_sw.val / (self.sw_width_sec / self.t_unit), self.area_vst.val, 0, 0))
            self.frac_users_new_vs_sw = self.ts_init(util.divide_treatzero(self.n_users_new_vs_sw.val, self.n_users.val, 0.5, 0.5))
            self.frac_users_new_vs_previoustime = self.ts_init(util.divide_treatzero(self.n_users_new_vs_previoustime.val, self.n_users.val, 0.5, 0.5))

            n_changes_sw = np.zeros(self.n_t_bins + 1)
            n_changes_cumsum = np.cumsum(self.n_changes.val)
            sw = self.sw_width
            n_changes_sw[0:sw] = n_changes_cumsum[0:sw]
            n_changes_sw[sw:] = n_changes_cumsum[sw:] - n_changes_cumsum[:(-sw)]
            self.changes_per_user_sw = self.ts_init(util.divide_treatzero(n_changes_sw, self.n_users_sw.val, 1, 1))

            self.frac_attack_changes = self.ts_init(util.divide_treatzero(self.n_changes.val - self.n_defense_changes.val, self.n_changes.val, 0.5, 0.5))
            self.frac_defenseonly_users = self.ts_init(util.divide_treatzero(self.n_defenseonly_users.val - self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5))
            self.frac_bothattdef_users = self.ts_init(util.divide_treatzero(self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5))
            self.frac_attackonly_users = self.ts_init(util.divide_treatzero(self.n_users.val - self.n_defenseonly_users.val - self.n_bothattdef_users.val, self.n_users.val, 0.5, 0.5))

        if self.compute_vars['stability'] > 0 and self.compute_vars['entropy'] > 0:
            self.frac_pixdiff_stable_vs_swref = self.ts_init(util.divide_treatzero(self.diff_pixels_stable_vs_swref.val, self.area_vst.val, 0, 0))
            self.frac_pixdiff_inst_vs_stable_norm = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_stable.val / self.t_norm, self.area_vst.val, 0, 0))

        if self.compute_vars['transitions'] > 0:
            self.frac_pixdiff_inst_vs_swref = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_swref.val, self.area_vst.val, 0, 0))
            self.frac_pixdiff_inst_vs_swref_forwardlook = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_swref_forwardlook.val, self.area_vst.val, 0, 0))

        if self.compute_vars['entropy'] > 0 or self.compute_vars['transitions'] > 1:
            self.frac_pixdiff_inst_vs_inst_norm = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_inst.val / self.t_norm, self.area_vst.val, 0, 0))
            self.frac_pixdiff_inst_vs_inst_downscaled2 = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_inst_downscaled2.val / self.t_norm, self.area_vst.val / 2**2, 0, 0))
            self.frac_pixdiff_inst_vs_inst_downscaled4 = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_inst_downscaled4.val / self.t_norm, self.area_vst.val / 4**2, 0, 0))
            self.frac_pixdiff_inst_vs_inst_downscaled16pix = self.ts_init(util.divide_treatzero(self.diff_pixels_inst_vs_inst_downscaled16pix.val / self.t_norm, self.area_vst.val / self.maxscale**2, 0, 0))
            self.entropy = self.ts_init(util.divide_treatzero(self.size_compressed.val, self.area_vst.val))
            self.entropy_bmpnorm = self.ts_init(util.divide_treatzero(self.size_compressed.val, self.size_uncompressed.val, 0, 0))
            self.entropy.val[0] = 0
            self.entropy_bmpnorm.val[0] = 0
            idx_dividebyzero = np.where(self.area_vst.val == 0)
            self.entropy.val[idx_dividebyzero] = self.entropy_bmpnorm.val[idx_dividebyzero] * 3.2
            self.wavelet_high_to_low = self.ts_init(util.divide_treatzero(self.wavelet_high_freq.val, self.wavelet_low_freq.val))
            self.wavelet_mid_to_low = self.ts_init(util.divide_treatzero(self.wavelet_mid_freq.val, self.wavelet_low_freq.val))
            self.wavelet_high_to_mid = self.ts_init(util.divide_treatzero(self.wavelet_high_freq.val, self.wavelet_mid_freq.val))
            self.wavelet_high_to_low_tm = self.ts_init(util.divide_treatzero(self.wavelet_high_freq_tm.val, self.wavelet_low_freq_tm.val))

        if self.compute_vars['stability'] > 1:
            self.entropy_stab_im = self.ts_init(util.divide_treatzero(self.size_compr_stab_im.val, self.area_vst.val))

        if self.compute_vars['clustering'] > 0 and self.ripley_norm is not None:
            if self.ripley_distances[0] == 2:
                self.ripley_norm[0] = self.ts_init(util.divide_treatzero(self.ripley_norm[0].val, np.sqrt(self.area_vst.val)))
            self.image_shift_minpos = self.ts_init(util.divide_treatzero(self.image_shift_minpos.val, np.sqrt(self.area_vst.val)))

        if self.compute_vars['other'] > 0:
            self.frac_moderator_changes = self.ts_init(util.divide_treatzero(self.n_moderator_changes.val, self.n_changes.val, 0, 0))
            self.frac_cooldowncheat_changes = self.ts_init(util.divide_treatzero(self.n_cooldowncheat_changes.val, self.n_changes.val, 0, 0))
            self.frac_redundant_color_changes = self.ts_init(util.divide_treatzero(self.n_redundant_color_changes.val, self.n_changes.val, 0, 0))
            self.frac_redundant_coloranduser_changes = self.ts_init(util.divide_treatzero(self.n_redundant_coloranduser_changes.val, self.n_changes.val, 0, 0))

    def search_transitions(self, cpart):
        par = self.transition_param
        self.frac_pixdiff_inst_vs_swref.set_ratio_to_sw_average()
        self.frac_pixdiff_inst_vs_swref_forwardlook.set_ratio_to_sw_average()
        transitions = tran.find_transitions(self.t_lims,
                                            self.frac_pixdiff_inst_vs_swref, self.frac_pixdiff_inst_vs_swref_forwardlook,
                                            tmin=self.tmin,
                                            cutoff_abs=par[0], cutoff_rel=par[1], 
                                            sw_width=self.sw_width_sec, stable_area_timeranges=self.stable_borders_timeranges)
        self.transition_tinds = transitions[0]
        self.transition_times = transitions[1]
        self.n_transitions = len(transitions[1])
        trans = self.compute_vars['transitions']

        self.refimage_pretrans = cpart.white_image(3, images_number=self.n_transitions)
        self.refimage_posttrans = cpart.white_image(3, images_number=self.n_transitions)
        self.refimage_intrans = cpart.white_image(3, images_number=self.n_transitions) if trans > 1 else None
        self.frac_diff_pixels_pre_vs_post_trans = np.empty(self.n_transitions)

        self.trans_start_time = np.empty(self.n_transitions)
        self.trans_start_tind = np.empty(self.n_transitions, dtype=np.int64)

        for j in range(0, self.n_transitions):
            self.trans_start_time[j] = tran.transition_start_time(self, j)[1] # take the mean here, but could be the median (among variables)
            self.trans_start_tind[j] = np.argmax(self.t_lims >= self.trans_start_time[j])

            beg_trans = min(self.transition_tinds[j][0], np.argmax(self.t_lims > var.TIME_GREYOUT))
            end_trans = min(self.transition_tinds[j][1] + self.sw_width, self.n_t_bins)
            active_coords = cpart.active_coord_inds(self.t_lims[beg_trans], self.t_lims[end_trans])
            self.frac_diff_pixels_pre_vs_post_trans[j] = np.count_nonzero(self.refimage_sw_flat[beg_trans][active_coords]
                                                                        - self.refimage_sw_flat[end_trans][active_coords]) / self.area

            if trans > 1:
                self.refimage_pretrans[j] = self.refimage_sw[beg_trans]
                self.refimage_posttrans[j] = self.refimage_sw[end_trans]
                self.refimage_intrans[j] = self.true_image[self.trans_start_tind[j]]

            if trans > 2:
                util.pixels_to_image(self.refimage_pretrans[j], cpart.out_name(), 'referenceimage_sw_pre_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_intrans[j], cpart.out_name(), 'referenceimage_at_transition'+str(j) + '.png')
                util.pixels_to_image(self.refimage_posttrans[j], cpart.out_name(), 'referenceimage_sw_post_transition'+str(j) + '.png')

    def fill_timeseries_info(self):
        def filepath(name):
            return os.path.join(var.FIGS_PATH, self.id, name+'.png')
        n_min_tunit = str(int(self.t_unit+1e-3)) + ' min'

        self.frac_pixdiff_inst_vs_stable_norm.desc_long = 'Fraction of pixels differing from the stable image in the previous step, normalized by the time unit.'
        self.frac_pixdiff_inst_vs_stable_norm.desc_short = 'frac of pixels differing from previous-step stable image / '+n_min_tunit
        self.frac_pixdiff_inst_vs_stable_norm.label = 'frac_pixdiff_inst_vs_stable'
        self.frac_pixdiff_inst_vs_stable_norm.savename = filepath('fraction_of_differing_pixels_vs_stable_normalized')

        self.frac_pixdiff_inst_vs_inst_norm.desc_long = 'Fraction of pixels differing from the instantaneous image in the previous step, normalized by the time unit.'
        self.frac_pixdiff_inst_vs_inst_norm.desc_short = 'frac of pixels differing from previous-step image / '+n_min_tunit
        self.frac_pixdiff_inst_vs_inst_norm.label = 'frac_pixdiff_inst_vs_inst'
        self.frac_pixdiff_inst_vs_inst_norm.savename = filepath('fraction_of_differing_pixels_normalized')

        self.frac_pixdiff_inst_vs_inst_downscaled2.desc_long = 'Fraction of pixels differing from the instantaneous image in the previous step, downscaled by factor 2.'
        self.frac_pixdiff_inst_vs_inst_downscaled2.desc_short = 'frac of pixels differing from previous-step image / '+n_min_tunit+'downscaled by 2.'
        self.frac_pixdiff_inst_vs_inst_downscaled2.label = 'frac_pixdiff_inst_vs_inst_downscaled2'
        self.frac_pixdiff_inst_vs_inst_downscaled2.savename = filepath('fraction_of_differing_pixels_downscaled2')

        self.frac_pixdiff_inst_vs_inst_downscaled4.desc_long = 'Fraction of pixels differing from the instantaneous image in the previous step, downscaled by factor 4.'
        self.frac_pixdiff_inst_vs_inst_downscaled4.desc_short = 'frac of pixels differing from previous-step image / '+n_min_tunit+'downscaled by 4.'
        self.frac_pixdiff_inst_vs_inst_downscaled4.label = 'frac_pixdiff_inst_vs_inst_downscaled4'
        self.frac_pixdiff_inst_vs_inst_downscaled4.savename = filepath('fraction_of_differing_pixels_downscaled4')

        self.frac_pixdiff_inst_vs_inst_downscaled16pix.desc_long = 'Fraction of pixels differing from the instantaneous image in the previous step, downscaled by factor 16.'
        self.frac_pixdiff_inst_vs_inst_downscaled16pix.desc_short = 'frac of pixels differing from previous-step image / '+n_min_tunit+'downscaled to 16 pixels.'
        self.frac_pixdiff_inst_vs_inst_downscaled16pix.label = 'frac_pixdiff_inst_vs_inst_downscaled16pix'
        self.frac_pixdiff_inst_vs_inst_downscaled16pix.savename = filepath('fraction_of_differing_pixels_downscaled16pix')

        self.frac_pixdiff_inst_vs_swref.desc_long = 'Fraction of pixels differing from the reference image in the preceding sliding window'
        self.frac_pixdiff_inst_vs_swref.desc_short = 'frac of pixels differing from sliding-window ref image'
        self.frac_pixdiff_inst_vs_swref.label = 'frac_pixdiff_inst_vs_swref'
        self.frac_pixdiff_inst_vs_swref.savename = filepath('fraction_of_differing_pixels_vs_slidingwindowref')

        self.frac_pixdiff_inst_vs_swref_forwardlook.desc_long = 'Fraction of pixels differing from the reference image in the future (forward-looking) sliding window'
        self.frac_pixdiff_inst_vs_swref_forwardlook.desc_short = 'frac of pixels differing from forward-looking sliding-window ref image'
        self.frac_pixdiff_inst_vs_swref_forwardlook.label = 'frac_pixdiff_inst_vs_swref_forwardlook'
        self.frac_pixdiff_inst_vs_swref_forwardlook.savename = filepath('fraction_of_differing_pixels_vs_slidingwindowref_forwardlook')

        self.n_users_norm.desc_long = 'number of users, normalized by the time unit and the number of active pixels.'
        self.n_users_norm.desc_short = '# users / area / '+n_min_tunit
        self.n_users_norm.label = 'n_users_norm'
        self.n_users_norm.savename = filepath('number_of_users_normalized')

        self.n_changes_norm.desc_long = 'number of pixel changes, normalized by the time unit and the number of active pixels.'
        self.n_changes_norm.desc_short = '# pixel changes / area / '+n_min_tunit
        self.n_changes_norm.label = 'n_changes_norm'
        self.n_changes_norm.savename = filepath('number_of_pixel_changes_normalized')

        self.entropy.desc_long = 'entropy, calculated as the computable information density of the instantaneous image'
        self.entropy.desc_short = 'entropy (computable information density)'
        self.entropy.label = 'entropy'
        self.entropy.savename = filepath('entropy')

        self.complexity_levenshtein.desc_long = 'Levenshtein-based spatial complexity CL score'
        self.complexity_levenshtein.desc_short = 'Levenshtein complexity CL'
        self.complexity_levenshtein.label = 'complexity_levenshtein'
        self.complexity_levenshtein.savename = filepath('complexity_levenshtein')

        self.complexity_multiscale.desc_long = 'normalized Zhang complexity K using mode-based coarse-graining'
        self.complexity_multiscale.desc_short = 'multiscale complexity K'
        self.complexity_multiscale.label = 'complexity_multiscale'
        self.complexity_multiscale.savename = filepath('complexity_multiscale')

        self.frac_attack_changes.desc_long = 'Fraction of the number of pixel changes that are attacking. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_attack_changes.desc_short = 'fraction of attack changes'
        self.frac_attack_changes.label = 'frac_attack_changes'
        self.frac_attack_changes.savename = filepath('fraction_attack_pixelchanges')

        self.frac_attackonly_users.desc_long = 'Fraction of the active users that are only attacking in this timestep. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_attackonly_users.desc_short = 'fraction of users only attacking'
        self.frac_attackonly_users.label = 'frac_attackonly_users'
        self.frac_attackonly_users.savename = filepath('fraction_of_users_onlyattacking')

        self.frac_defenseonly_users.desc_long = 'Fraction of the active users that are only defending in this timestep. \
            Defense is defined as following the stable reference image on a sliding window.'
        self.frac_defenseonly_users.desc_short = 'fraction of users only defending'
        self.frac_defenseonly_users.label = 'frac_defenseonly_users'
        self.frac_defenseonly_users.savename = filepath('fraction_of_users_onlydefending')

        self.frac_bothattdef_users.desc_long = 'Fraction of the active users that are both attacking and defending in this timestep. \
            Attack is defined compared to the stable reference image on a sliding window.'
        self.frac_bothattdef_users.desc_short = 'fraction of users both attacking and defending'
        self.frac_bothattdef_users.label = 'frac_bothattdef_users'
        self.frac_bothattdef_users.savename = filepath('fraction_of_users_bothattackingdefending')

        # same for wavelet_high_to_low, the ratio of wavelet high frequency to low frequency
        self.wavelet_high_to_low.desc_long = 'Ratio of image spatial wavelet high frequency to low frequency.'
        self.wavelet_high_to_low.desc_short = 'wavelet high to low frequency ratio'
        self.wavelet_high_to_low.label = 'wavelet_high_to_low'
        self.wavelet_high_to_low.savename = filepath('wavelet_high_to_low')

        self.wavelet_mid_to_low.desc_long = 'Ratio of image spatial wavelet mid frequency to low frequency.'
        self.wavelet_mid_to_low.desc_short = 'wavelet mid to low frequency ratio'
        self.wavelet_mid_to_low.label = 'wavelet_mid_to_low'
        self.wavelet_mid_to_low.savename = filepath('wavelet_mid_to_low')

        self.wavelet_high_to_mid.desc_long = 'Ratio of image spatial wavelet high frequency to mid frequency.'
        self.wavelet_high_to_mid.desc_short = 'wavelet high to mid frequency ratio'
        self.wavelet_high_to_mid.label = 'wavelet_high_to_mid'
        self.wavelet_high_to_mid.savename = filepath('wavelet_high_to_mid')

        self.wavelet_high_to_low_tm.desc_long = 'Ratio of temporal wavelet high frequency to low frequency.'
        self.wavelet_high_to_low_tm.desc_short = 'wavelet high to low frequency ratio (temporal)'
        self.wavelet_high_to_low_tm.label = 'wavelet_high_to_low_tm'
        self.wavelet_high_to_low_tm.savename = filepath('wavelet_high_to_low_tm')

        if self.compute_vars['attackdefense'] > 0:
            self.returnrate.desc_long ='fraction of attacked pixels that recover within 5 minutes'
            self.returnrate.desc_short = 'return rate (fraction of recovered pixels)'
            self.returnrate.label = 'returnrate'
            self.returnrate.savename = filepath('returnrate')

        desclong = ["Mean", "Median", "90th percentile", "Mean of highest decile"]
        descshort = ["mean", "median", "percentile90", "top decile mean"]
        labs = ["mean", "median", "percentile90", "meantopdecile"]

        for k in range(0,4):
            if self.compute_vars['attackdefense'] > 0:
                self.returntime[k].desc_long = desclong[k]+' time for pixels to recover from attack [s] / ln(2)'
                self.returntime[k].desc_short = descshort[k]+' pixel recovery time from attack [s] / ln(2)'
                self.returntime[k].label = 'returntime_'+labs[k]
                self.returntime[k].savename = filepath(labs[k]+'_pixel_recovery_time')
                
            self.instability_norm[k].desc_long = 'instability = (1 - stability), normalized by the time unit. \
                Stability is the time fraction that each pixel spent in its dominant color.'+desclong[k]+'of all active pixels'
            self.instability_norm[k].desc_short = 'instability '+descshort[k]+' / '+n_min_tunit
            self.instability_norm[k].label = 'instability_'+labs[k]
            self.instability_norm[k].savename = filepath(labs[k]+'_instability_normalized')

            self.runnerup_timeratio[k].desc_long = 'Ratio of time spent in the second most used color to that in the most used color, for each pixel. '+desclong[k]+' of all active pixels'
            self.runnerup_timeratio[k].desc_short = 't(runner-up color) / t(most stable color) ['+descshort[k]+']'
            self.runnerup_timeratio[k].label = 'runnerup_timeratio_'+labs[k]
            self.runnerup_timeratio[k].savename = filepath(labs[k]+'_runnerup_timeratio')

            self.n_used_colors[k].desc_long = 'Number of used colors in each pixel. '+desclong[k]+' of all active pixels'
            self.n_used_colors[k].desc_short = '# used colors / pixel ['+descshort[k]+']'
            self.n_used_colors[k].label = 'n_colors_'+labs[k]
            self.n_used_colors[k].savename = filepath(labs[k]+'_n_used_colors')

        if self.compute_vars['clustering'] > 0:
            dist = [str(self.ripley_distances[0]), '20%% of size', '50%% of size']
            for k in range(0,len(self.ripley_distances)):
                if self.compute_vars['clustering'] > 1:
                    self.ripley[k].desc_long = 'Ripley\'s K function, at distance ' + dist[k]  
                    self.ripley[k].desc_short = 'Ripley\'s K function [d=' + dist[k] + ']'
                    self.ripley[k].label = 'ripley_d='+dist[k]
                    self.ripley[k].savename = filepath('ripley_kfunction_d'+dist[k])

                self.ripley_norm[k].desc_long = 'Ripley\'s K function, at distance ' + dist[k] + 'normalized to randomized pixel changes positions and to size'
                self.ripley_norm[k].desc_short = 'Ripley\'s K function [d=' + dist[k] + '] normalized to randomized'
                self.ripley_norm[k].label = 'ripley_norm_d='+dist[k]
                self.ripley_norm[k].savename = filepath('ripley_k_norm_d'+dist[k])

            # same for crosscorr
            if self.compute_vars['clustering'] > 1:
                for k in range(0, len(self.crosscorr_distances)):
                    self.crosscorr[k].desc_long = 'Cross-correlation between t and t-1 images at shift ' + str(self.crosscorr_distances[k])
                    self.crosscorr[k].desc_short = 'Cross-correlation t vs t-1 [' + str(self.crosscorr_distances[k]) + ']'
                    self.crosscorr[k].label = 'crosscorr_d='+str(self.crosscorr_distances[k])
                    self.crosscorr[k].savename = filepath('crosscorr_d'+str(self.crosscorr_distances[k]))

            self.image_shift_min.desc_long = 'Minimum of normalized cross-correlation (shift) between t and t-1 images'
            self.image_shift_min.desc_short = 'min value cross-correlation shift t vs t-1'
            self.image_shift_min.label = 'image_shift_min'
            self.image_shift_min.savename = filepath('image_shift_min')

            self.image_shift_minpos.desc_long = 'Shift at the minimum normalized cross-correlation between t and t-1 images'
            self.image_shift_minpos.desc_short = 'Shift at min value cross-correlation shift t vs t-1'
            self.image_shift_minpos.label = 'image_shift_minpos'
            self.image_shift_minpos.savename = filepath('image_shift_minpos')

            self.image_shift_slope.desc_long = 'Slope of the linear fit to the normalized cross-correlation between t and t-1 images'
            self.image_shift_slope.desc_short = 'Slope of linear fit to cross-correlation (shift) t vs t-1'
            self.image_shift_slope.label = 'image_shift_slope'
            self.image_shift_slope.savename = filepath('image_shift_slope')

            if self.compute_vars['clustering'] > 1:
                self.dist_average.desc_long = 'Average distance between all pairs of pixel changes'
                self.dist_average.desc_short = 'Average distance between pix changes'
                self.dist_average.label = 'distance_between_changes'
                self.dist_average.savename = filepath('distance_between_changes')

            self.dist_average_norm.desc_long = 'Average distance between all pairs of pixel changes, normalized to randomized pixel changes positions and to size'
            self.dist_average_norm.desc_short = 'Average distance between pix changes, norm to random'
            self.dist_average_norm.label = 'distance_between_changes_norm'
            self.dist_average_norm.savename = filepath('distance_between_changes_norm')

        self.cumul_attack_timefrac.desc_long = 'Fraction of the time that all pixels spent in an attack color [s]'
        self.cumul_attack_timefrac.desc_short = 'frac of time spent in attack colors [s]'
        self.cumul_attack_timefrac.label = 'cumulated_attack_timefrac'
        self.cumul_attack_timefrac.savename = filepath('attack_time_fraction_allpixels')

        self.frac_moderator_changes.desc_long = 'Fraction of active pixels changes issued from moderators'
        self.frac_moderator_changes.desc_short = 'fraction of moderator changes'
        self.frac_moderator_changes.label = 'frac_mod_changes'
        self.frac_moderator_changes.savename = filepath('fraction_moderator_changes')

        self.frac_cooldowncheat_changes.desc_long = 'Fraction of active pixels changes that break the 5-minute per-user cooldown'
        self.frac_cooldowncheat_changes.desc_short = 'fraction of cooldown-cheating changes'
        self.frac_cooldowncheat_changes.label = 'frac_cooldowncheat_changes'
        self.frac_cooldowncheat_changes.savename = filepath('fraction_cooldowncheating_changes')

        self.frac_redundant_color_changes.desc_long = 'Fraction of active pixels changes that are redundant in color with the pixel content before its change'
        self.frac_redundant_color_changes.desc_short = 'fraction of redundant changes'
        self.frac_redundant_color_changes.label = 'frac_redundant_changes'
        self.frac_redundant_color_changes.savename = filepath('fraction_redundant_changes')

        self.frac_redundant_coloranduser_changes.desc_long = 'Fraction of active pixels changes that are redundant in color and user with the pixel content before its change'
        self.frac_redundant_coloranduser_changes.desc_short = 'fraction of redundant (color and user) changes'
        self.frac_redundant_coloranduser_changes.savename = filepath('fraction_redundant_coloranduser_changes')
        self.frac_redundant_coloranduser_changes.label = 'frac_redundant_coloranduser_changes'

        self.changes_per_user_sw.desc_long = 'Changes per unique user active over the preceding sliding window'
        self.changes_per_user_sw.desc_short = 'Changes per user (sliding window)'
        self.changes_per_user_sw.savename = filepath('changes_per_user_slidingwindow')
        self.changes_per_user_sw.label = 'changes_per_user_sw'

        self.n_users_sw_norm.desc_long = 'Number of unique users active over the preceding sliding window'
        self.n_users_sw_norm.desc_short = '# users (sliding window)'
        self.n_users_sw_norm.savename = filepath('number_users_slidingwindow')
        self.n_users_sw_norm.label = 'n_users_sw'

        self.frac_users_new_vs_sw.desc_long = 'Fraction of new users in this timestep compared to those active in the preceding sliding window'
        self.frac_users_new_vs_sw.desc_short = 'Fraction of new users vs sliding window'
        self.frac_users_new_vs_sw.savename = filepath('fraction_new_users_vs_slidingwindow')
        self.frac_users_new_vs_sw.label = 'frac_new_users_vs_sw'

        self.frac_users_new_vs_previoustime.desc_long = 'Fraction of new users in this timestep compared to those active in the previous timestep'
        self.frac_users_new_vs_previoustime.desc_short = 'Fraction of new users vs previous time'
        self.frac_users_new_vs_previoustime.savename = filepath('frac_new_users_vs_previoustime')
        self.frac_users_new_vs_previoustime.label = 'frac_new_users_vs_previoustime'

        self.fractal_dim_weighted.desc_long = 'Fractal dimension, averaged over colors weighted by their abundance'
        self.fractal_dim_weighted.desc_short = 'Fraction dimension (color-weighted)'
        self.fractal_dim_weighted.savename = filepath('fractal_dimension_weighted')
        self.fractal_dim_weighted.label = 'fractal_dimension'

        #self.variance.desc_long = 'Variance'
        #self.variance.desc_short = 'Variance'
        #self.variance.savename = filepath('variance')
        #self.variance.label = 'variance'

        self.variance2.desc_long = 'Variance in parallel over all pixels'
        self.variance2.desc_short = 'Variance all pixels'
        self.variance2.savename = filepath('variance_allpixels')
        self.variance2.label = 'variance_all_pixels'

        self.variance_multinom.desc_long = 'Variance based on multinomial distribution'
        self.variance_multinom.desc_short = 'Variance (multinomial)'
        self.variance_multinom.savename = filepath('variance_multinomial')
        self.variance_multinom.label = 'variance_multinom'

        self.variance_subdom.desc_long = 'Variance based on subdominant dot product'
        self.variance_subdom.desc_short = 'Variance (subdominant dot product)'
        self.variance_subdom.savename = filepath('variance_subdominant')
        self.variance_subdom.label = 'variance_subdom'

        self.variance_from_frac_pixdiff_inst.desc_long = 'Variance based on squares of frac_pixdiff_inst_vs_inst'
        self.variance_from_frac_pixdiff_inst.desc_short = 'Variance (frac_pixdiff_inst)'
        self.variance_from_frac_pixdiff_inst.savename = filepath('variance_frac_pixdiff_inst')
        self.variance_from_frac_pixdiff_inst.label = 'variance_frac_pixdiff_inst'
        
        self.autocorr_bycase.desc_long = 'Autocorrelation by case'
        self.autocorr_bycase.desc_short = 'Autocorrelation by case'
        self.autocorr_bycase.savename = filepath('autocorr_bycase')
        self.autocorr_bycase.label = 'autocorr_bycase'

        self.autocorr_bycase_norm.desc_long = 'Autocorrelation by case, over all pixels'
        self.autocorr_bycase_norm.desc_short = 'autocorrelation_bycase_allpixels'
        self.autocorr_bycase_norm.savename = filepath('autocorr_bycase_norm')
        self.autocorr_bycase_norm.label = 'autocorr_bycase_norm'

        self.autocorr_dissimil.desc_long = 'Autocorrelation by dissimilarity'
        self.autocorr_dissimil.desc_short = 'autocorrelation_dissimil'
        self.autocorr_dissimil.savename = filepath('autocorr_dissimil')
        self.autocorr_dissimil.label = 'autocorr_dissimil'

        self.autocorr_subdom.desc_long = 'Autocorrelation based on subdominant dot product'
        self.autocorr_subdom.desc_short = 'autocorrelation_subdominant'
        self.autocorr_subdom.savename = filepath('autocorr_subdom')
        self.autocorr_subdom.label = 'autocorr_subdom'

        self.autocorr_multinom.desc_long = 'Autocorrelation based on multinomial distribution'
        self.autocorr_multinom.desc_short = 'autocorrelation_multinomial'
        self.autocorr_multinom.savename = filepath('autocorr_multinom')
        self.autocorr_multinom.label = 'autocorr_multinom'

        if self.compute_vars['inoutgroup'] > 1:
            self.ingroup_users_vst_desc_short = 'In-group users in this time step'
            self.ingroup_users_vst_desc_long = 'Unique user IDs with at least one in-group change in each time step.'
            self.outgroup_users_vst_desc_short = 'Out-group users in this time step'
            self.outgroup_users_vst_desc_long = 'Unique user IDs with at least one out-group change in each time step.'

        #self.autocorr.desc_long = 'autocorr'
        #self.autocorr.desc_short = 'autocorr'
        #self.autocorr.savename = filepath('autocorr')
        #self.autocorr.label = 'autocorr'

        #self.autocorr2.desc_long = 'autocorrelation over all pixels'
        #self.autocorr2.desc_short = 'autocorrelation_allpixels'
        #self.autocorr2.savename = filepath('autocorrelation_allpixels')
        #self.autocorr2.label = 'autocorrelation_allpixels'
