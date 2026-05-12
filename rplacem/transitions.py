import numpy as np
import rplacem.canvas_part as cp
import rplacem.compute_variables as comp
import rplacem.utilities as util
from rplacem import var as var
import math
from scipy import optimize

def limits_sequence_of_true(cond, min_len_seq):
    '''
    Gives the limits (indices) of sequences of True values from a boolean array

    parameters
    ----------
    cond: boolean array
        array of conditions that needs to be true between the beginning and end indices that are output
    min_len_seq: int
        minimum length of the sequences that will be kept

    returns
    -------
    beg
    end: int array of length n_sequences_output
        contains the beginning/end indices of wanted sequences
    '''
    cond_fenced = np.concatenate(([0], cond, [0]))
    lim_seq = np.flatnonzero(np.diff(cond_fenced) != 0) # limits of sequences of "True"
    beg_seq_true = lim_seq[::2] # beginning of sequences
    end_seq_true = lim_seq[1::2] # ends of sequences

    # keep only long enough sequences
    if min_len_seq > 1:
        long_enough = (end_seq_true - beg_seq_true >= min_len_seq)
        beg_seq_true = beg_seq_true[long_enough]
        end_seq_true = end_seq_true[long_enough]

    return np.column_stack((beg_seq_true, end_seq_true))

def merge_similar_transitions(transitions, require_same_posttrans=False):
    '''
    Merge the transitions if they have the same pre- (and optionally post-) transition periods 
    '''
    deleted_idx = []
    trans_merged = []
     
    for i in range(0, len(transitions)):
        if i in deleted_idx: # this transition was previously deleted in the second-level loop
            continue
        trans = transitions[i]
        for j in range(i, len(transitions)):
            trans2 = transitions[j]
            if (trans[0] == trans2[0] and trans[1] == trans2[1] 
                and ((not require_same_posttrans) or 
                     (trans[4] == trans2[4] and trans[5] == trans2[5]))):
                trans[2] = min(trans[2], trans2[2]) # new transition time interval is the smallest that contains both transition intervals
                trans[3] = max(trans[3], trans2[3])
                if not require_same_posttrans:
                    trans[4] = max(trans[4], trans2[4])
                    trans[5] = max(trans[5], trans2[5])
                deleted_idx.append(j)
        trans_merged.append(trans)

    return trans_merged

def merge_close_transitions(transitions, sw_width):
    '''
    Merge the transitions if they are closer than sw_width to each other 
    '''
    deleted_idx = []
    trans_merged = []
     
    for i in range(0, len(transitions)):
        if i in deleted_idx: # this transition was previously deleted in the second-level loop
            continue
        trans = transitions[i]
        for j in range(i, len(transitions)):
            trans2 = transitions[j]
            if trans2[0] <= trans[1] + sw_width:
                trans[1] = max(trans[1], trans2[1])
                deleted_idx.append(j)
        trans_merged.append(trans)

    return trans_merged

def find_transitions(t_lims, 
                     testvar_pre, testvar_post,
                     tmin=0,
                     cutoff_abs=0.25, 
                     cutoff_rel=6,
                     sw_width=3*3600,
                     stable_area_timeranges=[[0, var.TIME_TOTAL]]
                     ):
    '''
    identifies transitions in a cpart.
    
    parameters
    ----------
    t_lims : 1d array-like of floats
        time intervals in which the pixel states are studied. Must be ordered in a crescent way, start at 0, and be regularly spaced
    testvar_pre : 1d array-like of floats
        The variable on which cutoffs will be applied to detect transitions
    cutoff_abs : float
        the lower cutoff on instability to define when a transition is happening
    cutoff_rel : float
        the higher cutoff on instability, needs to be not exceeded before and after the potential transition, for a validated transition

    returns
    -------
    (full_transition, full_transition_times) :
        full_transition is a 2d array. 
        full_transition[i] contains, for the found transition #i, the following size-6 array (of indices of the input instability array) :
            [beginning of preceding stable region, end of stable, 
            beginning of transition region, end of transition, 
            beginning of subsequent stable region, end of stable]
        (NB: a region of indices [n, m] includes all time intervals from n to m included)
        full_transition_times is the same array but with the times corresponding to those indices 
    '''
    # preliminary
    if cutoff_rel < 1.01:
        raise ValueError("ERROR in find_transitions(): relative cutoff for transitions must be higher than 1")
    testvar_pre_abs = testvar_pre.val
    testvar_pre_rel = testvar_pre.ratio_to_sw_mean
    testvar_post_abs = testvar_post.val
    testvar_post_rel = testvar_post.ratio_to_sw_mean

    # transition and pre and post-transition stability conditions
    intrans_cond = np.array((testvar_pre_abs > cutoff_abs) & (testvar_pre_rel > cutoff_rel) 
                          & (t_lims >= tmin + sw_width))
    not_at_borderpath_change = np.zeros(t_lims.shape, dtype=bool)
    for i_range in range(len(stable_area_timeranges)):
        not_at_borderpath_change = np.logical_or(not_at_borderpath_change, 
                                                 np.array((t_lims >= stable_area_timeranges[i_range, 0] + sw_width) & (t_lims <= stable_area_timeranges[i_range, 1])))
    intrans_cond = np.logical_and(intrans_cond, not_at_borderpath_change)
    pasttrans_cond = np.array((testvar_post_abs < cutoff_abs) | (testvar_post_rel < 1/cutoff_rel))

    # get the beg and end of sequences of at least len_stableregion indices that pass conditions
    seq_intrans = limits_sequence_of_true(intrans_cond, 1)
    seq_pasttrans = limits_sequence_of_true(pasttrans_cond, 1)

    # keep only transitions that are surrounded by close-enough stable periods
    full_transition_tmp = []
    for tr in seq_intrans:
        if t_lims[min(tr[0]+1, len(t_lims)-1)] >= var.TIME_GREYOUT: # exclude the arrival of grey- or white-only pixelchanges from transitions
            continue
        
        posttrans_stable_seq = np.where((seq_pasttrans[:, 0] >= tr[0])
                                      & (seq_pasttrans[:, 0] <= tr[0] + sw_width))[0]
        if len(posttrans_stable_seq) > 0:
            endtrans = max(seq_pasttrans[posttrans_stable_seq[0]][0], tr[1]) # max of the transition-end given by the pre- and post-transition variables
        else:
            endtrans = min(tr[1], tr[0] + sw_width) # maximum sw_width duration of the transition

        full_transition_tmp.append(np.array([tr[0], endtrans]))
    
    # merge transitions that have the same preceding and subsequent stable regions
    full_transitions = merge_close_transitions(np.array(full_transition_tmp), int(sw_width/(t_lims[1]-t_lims[0])))
    
    full_transitions_cap = np.copy(full_transitions)
    full_transitions_cap[full_transitions_cap > (len(t_lims)-1)] = len(t_lims)-1
    return (np.array(full_transitions), t_lims[np.array(full_transitions_cap, dtype=int)])

def transition_start_time(cpstat, tr):

    vars = [cpstat.n_changes_norm.val,
            #cpstat.frac_attack_changes.val - 0.5,
            #cpstat.returntime_median_overln2.val,
            cpstat.frac_pixdiff_inst_vs_swref.val,
            #cpstat.frac_pixdiff_stable_vs_swref.val,
            #1 - cpstat.frac_defenseonly_users.val - cpstat.frac_bothattdef_users.val - 0.5, #attackonly - 0.5
            ]
    if cpstat.compute_vars['attackdefense'] > 0: 
        vars.append(cpstat.cumul_attack_timefrac.val)
    if cpstat.compute_vars['stability'] > 0:
        vars.append(cpstat.instability_norm[0].val)
        vars.append(cpstat.frac_pixdiff_inst_vs_stable_norm.val)

    halfmax_times = np.zeros(len(vars))

    t_lims = cpstat.t_lims
    # indices of the transition times
    trans_tind = cpstat.transition_tinds[tr]
    ind_st = trans_tind[0] - 1
    ind_end = trans_tind[1] + 1

    for i in range(len(vars)):
        # max of vars over transition time range
        maxi = np.max(vars[i][ind_st : ind_end])
        # look for the time at which vars reaches halfmax value, starting search from end of pre-trans stable period
        halfmax_ind = ind_st + np.argmax(vars[i][ ind_st:ind_end ] > maxi/2)
        # use a linear interpolation of the values of the variable at each time step
        lin_interpol = lambda x: np.interp(x, t_lims[np.arange(halfmax_ind-1, halfmax_ind+1)], 
                                              vars[i][ (halfmax_ind-1):(halfmax_ind+1) ]  ) - maxi/2
        halfmax_times[i] = optimize.fsolve(lin_interpol, [t_lims[halfmax_ind-1]], xtol=5e-4)[0]
        #print(i, halfmax_times[i])

    mean_halfmaxtime = np.mean(halfmax_times)
    median_halfmaxtime = np.median(halfmax_times)
    rms_halfmaxtime = np.std(halfmax_times)
    #print(mean_halfmaxtime, rms_halfmaxtime, median_halfmaxtime)
    #print(cpstat.transition_times[tr][2:4])

    return halfmax_times, mean_halfmaxtime, median_halfmaxtime, rms_halfmaxtime


def transition_start_time_simple(cpstat):
    thresvar_abs = cpstat.frac_pixdiff_inst_vs_swref.val
    thresvar_rel = cpstat.frac_pixdiff_inst_vs_swref.ratio_to_sw_mean
    tlims = cpstat.t_lims
    starttimes = []

    for tr in np.arange(0, cpstat.n_transitions):
        # indices of the transition times
        trans_tind = cpstat.transition_tinds[tr]
        ind_st = trans_tind[0]
        
        # linear interpolation to find the time at which the variable is exactly equal to the transition threshold
        boundary_vals = [thresvar_abs[ind_st - 1], thresvar_abs[ind_st]] # values of variable right before and after the beginning idx of the transition
        boundary_times = [tlims[ind_st-1], tlims[ind_st]]
        search_val = cpstat.transition_param[0]
        start_t_abs = np.interp(search_val, boundary_vals, boundary_times)
        # same for relative threshold
        boundary_vals = [thresvar_rel[ind_st - 1], thresvar_rel[ind_st]] # values of variable right before and after the beginning idx of the transition
        boundary_times = [tlims[ind_st-1], tlims[ind_st]]
        search_val = cpstat.transition_param[1]
        start_t_rel = np.interp(search_val, boundary_vals, boundary_times)

        starttimes.append(max(start_t_abs, start_t_rel))

    return np.array(starttimes)