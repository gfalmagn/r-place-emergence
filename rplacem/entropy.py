import numpy as np
import zlib
import math
import pywt
from hilbertcurve.hilbertcurve import HilbertCurve
from numba import jit
from Levenshtein import distance as levenshtein_distance
from skimage.util.shape import view_as_blocks as _view_as_blocks
import scipy
from rplacem import var as var
import os
import pickle
import collections
import warnings


def get_hilbert_mask(lattice_boxv):
    side_x, side_y = np.asarray(lattice_boxv, dtype=int)
    side = 2 ** math.ceil(math.log2(max(side_x, side_y)))
    order = int(math.log2(side))
    hilbert_curve = HilbertCurve(order, 2)
    indices = []
    for y in range(side):
        for x in range(side):
            indices.append((hilbert_curve.distance_from_point([x, y]), y * side + x))
    indices.sort(key=lambda item: item[0])
    return np.asarray([idx for _, idx in indices], dtype=np.int64)


def _compressed_length(data):
    return len(zlib.compress(np.asarray(data, dtype=np.uint8).tobytes()))


def lempel_ziv_complexity(x, method='lz77'):
    compressed_length = _compressed_length(x)
    if method == 'lz77':
        return compressed_length, compressed_length
    if method == 'lz78':
        return compressed_length
    raise NotImplementedError


def _lz77_complexity(x):
    res = lempel_ziv_complexity(x, 'lz77')
    if isinstance(res, tuple):
        return res
    return res, res


def _lz78_complexity(x):
    res = lempel_ziv_complexity(x, 'lz78')
    if isinstance(res, tuple):
        return res[0]
    return res


def get_comp_size_bytes(x, algorithm='deflate', **kwargs):
    compressed_length = _compressed_length(x)
    return compressed_length, None, None


def block_entropy(x, *args, **kwargs):
    values = np.asarray(x).ravel()
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def calc_size(pixels):
    '''
    Calculates the size of the uncompressed pixels
    '''
    data = pixels.tobytes()
    len_data = len(data)
    return len_data


def calc_compressed_size(pixels, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    Calculates the compressed file size of an image

    Parameters
    ---------
    pixels: numpy array
        2d pixel info
    flattening: string, optional
        Type of flattening used. if "hilbert_sweetsourcod", uses the hilbert curve flattening
        from the sweetsourcod package developed by Stefan Martiniani, where the functions we use are copied in this project. 
        if "ravel", then flattens into a simple 1d array without taking advantage of the original 2d structure of the image.
        "hilbert" is another implementation of the hilbert curve method.

    Returns
    -------
    len_compressed: float
        Length of the compressed image array.

    '''
    if flattening[0:7] == 'hilbert':
        # for any hilbert method, have to set up pixel padding and define exponent of 2
        power2 = math.ceil(math.log2(max(pixels.shape[0], pixels.shape[1])))
        pixels_pad = np.zeros((2**power2, 2**power2))
        pixels_pad[:pixels.shape[0], :pixels.shape[1]] = pixels

    if flattening == 'hilbert_sweetsourcod':
        lattice_boxv = np.asarray(pixels_pad.shape[::-1])
        hilbert_mask = get_hilbert_mask(lattice_boxv)
        pixels_flat = mask_array(pixels_pad.ravel(), hilbert_mask).astype('uint8')

    if flattening == 'hilbert_pkg':
        pixels_flat = np.zeros(len(pixels_pad.flatten()))
        hc = HilbertCurve(power2, 2)
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                h_index = hc.distance_from_point([i, j])
                pixels_flat[h_index] = pixels[i, j]

    elif flattening == 'ravel':
        pixels_flat = pixels.flatten()

    if compression == 'LZ77':
        len_compressed, _ = _lz77_complexity(pixels_flat)

    if compression == 'LZ78':
        len_compressed = _lz78_complexity(pixels_flat)

    if compression == 'DEFLATE':
        data = pixels_flat.tobytes()
        compressed_data = zlib.compress(data)
        len_compressed = len(compressed_data)

    return len_compressed

def normalize_entropy(entropy, area, time, 
                      compression="LZ77", flattening="ravel", num_iter=10, len_max=1000, minmax_entropy=None):
    '''
    Normalize the entropy
    '''
    (f_entropy_min, 
    f_entropy_max_8,
    f_entropy_max_16, 
    f_entropy_max_24, 
    f_entropy_max_32) = calc_min_max_entropy(compression=compression, 
                                             flattening=flattening,
                                             num_iter=num_iter,
                                             len_max=len_max)
    if time < var.TIME_16COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_8(area) - f_entropy_min(area))
    if time >= var.TIME_16COLORS and time < var.TIME_24COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_16(area) - f_entropy_min(area))
    elif time >= var.TIME_24COLORS and time < var.TIME_32COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_24(area) - f_entropy_min(area))
    elif time >= var.TIME_32COLORS:
        entropy_norm = (entropy - f_entropy_min(area)) / (f_entropy_max_32(area) - f_entropy_min(area))

    return entropy_norm

def calc_min_max_entropy(compression="LZ77",
                         flattening="ravel",
                         num_iter=10,
                         len_max=1000):
    """
    Calculate the min and max entropy for 16, 24, and 32 colors
    if the functions are already saved in a pickle file, load them
    """
    files = os.listdir(var.DATA_PATH)
    if 'entropy_min_max.pkl' in files:
        with open(os.path.join(var.DATA_PATH, 'entropy_min_max.pkl'), 'rb') as f:
            f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32 = pickle.load(f)
    
    else:
        # define array sequences for each length, using 16, 32, or all black
        squares = np.arange(1, len_max)**2
        entropy_8 = np.zeros(len(squares))
        entropy_16 = np.zeros(len(squares))
        entropy_24 = np.zeros(len(squares))
        entropy_32 = np.zeros(len(squares))

        for i in range(0, len(squares)):
            sq_len = squares[i]

            flat_black = np.zeros(sq_len)
            flat_black = flat_black.reshape(int(np.sqrt(sq_len)),
                                            int(np.sqrt(sq_len)))
            entropy_8_range = np.zeros(num_iter)
            entropy_16_range = np.zeros(num_iter)
            entropy_24_range = np.zeros(num_iter)
            entropy_32_range = np.zeros(num_iter)
            # entropy_black_range = np.zeros(num_iter)

            for j in range(num_iter):
                # entropy 8
                rand_flat_8 = np.random.choice(8, size=sq_len)
                rand_flat_8= rand_flat_8.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_8,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_8_range[j] = len_comp / sq_len

                # entropy 16
                rand_flat_16 = np.random.choice(16, size=sq_len)
                rand_flat_16 = rand_flat_16.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_16,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_16_range[j] = len_comp / sq_len

                # entropy 24
                rand_flat_24 = np.random.choice(24, size=sq_len)
                rand_flat_24 = rand_flat_24.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_24,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_24_range[j] = len_comp / sq_len

                # entropy 32
                rand_flat_32 = np.random.choice(32, size=sq_len)
                rand_flat_32 = rand_flat_32.reshape(int(np.sqrt(sq_len)),
                                                    int(np.sqrt(sq_len)))
                len_comp = calc_compressed_size(rand_flat_32,
                                                    flattening=flattening,
                                                    compression=compression)
                entropy_32_range[j] = len_comp / sq_len

                # len_comp = ent.calc_compressed_size(flat_black, flattening=flattening, compression=compression)
                # entropy_black_range[j] = len_comp/sq_len
            entropy_8[i] = np.mean(entropy_8_range)
            entropy_16[i] = np.mean(entropy_16_range)
            entropy_24[i] = np.mean(entropy_24_range)
            entropy_32[i] = np.mean(entropy_32_range)

        f_entropy_min = scipy.interpolate.interp1d(squares,
                                                1 / squares,
                                                kind="linear")
        f_entropy_max_8 = scipy.interpolate.interp1d(squares,
                                                    entropy_8,
                                                    kind="linear")
        f_entropy_max_16 = scipy.interpolate.interp1d(squares,
                                                    entropy_16,
                                                    kind="linear")
        f_entropy_max_24 = scipy.interpolate.interp1d(squares,
                                                    entropy_24,
                                                    kind="linear")
        f_entropy_max_32 = scipy.interpolate.interp1d(squares,
                                                    entropy_32,
                                                    kind="linear")
        # entropy_black[i] = np.mean(entropy_black_range)
        # flat_white = np.ones(sq_len, dtype='int32')
        # flat_white = flat_white.reshape(int(np.sqrt(sq_len)), int(np.sqrt(sq_len)))
        # len_comp = ent.calc_compressed_size(flat_white, flattening=flattening, compression=compression)
        # entropy_white[i] = len_comp/sq_lenm

        # save the functions to a pickle file
        with open(os.path.join(var.DATA_PATH, 'entropy_min_max.pkl'), 'wb') as f:
            pickle.dump((f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32), f)

    return f_entropy_min, f_entropy_max_8, f_entropy_max_16, f_entropy_max_24, f_entropy_max_32


def randomize(pixels):
    '''
    Randomizes the order pixels in an image
    '''
    # Get the size of the array
    size = pixels.size

    # Create a shuffled index array
    shuffled_idx = np.random.permutation(size)
    shuffled_pix = np.reshape(pixels.flat[shuffled_idx], pixels.shape)

    return shuffled_pix


def decimate(pixels, delta, dec_start=0):
    '''
    Decimate the image at interval delta
    '''
    if len(pixels.shape) == 1:
        pixel_dec = pixels[dec_start::delta]
    if len(pixels.shape) == 2:
        pixel_dec = pixels[dec_start::delta, dec_start::delta]
    return pixel_dec


def calc_Q(pixels, delta, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    calculate the Q value (scales with compressibility) for a given interval delta.
    '''
    kappa = np.zeros(delta)
    kappa_shuff = np.zeros(delta)
    for i in range(delta):  # loop over all possible decimation start points
        # Get the decimated pixels
        pixel_dec = decimate(pixels, delta, dec_start=i)

        # Calculate the compressed size of the decimated image
        kappa[i] = calc_compressed_size(pixel_dec, flattening=flattening, compression=compression)

        # Calcualte the compressed size of the randomly shuffled image
        shuffled_pix = randomize(pixel_dec)
        kappa_shuff[i] = calc_compressed_size(shuffled_pix, flattening=flattening, compression=compression)

    # Calculate the Q value
    Q = 1 - np.mean(kappa)/np.mean(kappa_shuff)
    return Q


def calc_Q_delta_vst(canvas_part_stat, flattening='hilbert_sweetsourcod', compression='LZ77'):
    '''
    Calculate the Q vs delta curve for each instantaneous image in the canvas_part_stat
    '''
    # Get the image pixels over time.
    pixels_vst = canvas_part_stat.true_image

    # Get the range of deltas.
    delta = np.arange(1, math.floor(pixels_vst.shape[1]/2)+1)

    # Calculate Q
    n_tlims = canvas_part_stat.n_t_bins + 1
    q = np.zeros((n_tlims, len(delta)))
    for j in range(n_tlims):  # loop through the times
        for i in range(len(delta)):  # loop through the deltas
            q[j, i] = calc_Q(pixels_vst[j], delta[i], flattening=flattening, compression=compression)

    return q, delta
        
def fast_mode(x):
    '''
    Fast mode computation using bincount for 1D array of non-negative integers.
    Parameters:
        x (np.ndarray): 1D array of non-negative integers.
    Returns:
        int: The mode of the array (most frequent value).
    '''
    return np.argmax(np.bincount(x))

def mode_downsample(image, factor):
    """
    Downsample a 2D image by a given factor using mode-based coarse-graining.
    Handles incomplete blocks at edges by computing mode of available pixels.
    Uses efficient view_as_blocks for main grid and manual loops only for edges.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array of shape (H, W) with non-negative integers
    factor : int
        Downsampling factor (block size)
    
    Returns:
    --------
    np.ndarray
        Downsampled 2D array of shape (ceil(H/factor), ceil(W/factor))
    """
    H, W = image.shape
    
    # Calculate output dimensions and main grid size
    out_h = (H + factor - 1) // factor  # ceil(H / factor)
    out_w = (W + factor - 1) // factor  # ceil(W / factor)
    
    # Main grid dimensions (complete blocks only)
    main_h, main_w = H // factor, W // factor
    
    # Create output array
    modes = np.zeros((out_h, out_w), dtype=image.dtype)
    
    # Process main grid efficiently using view_as_blocks
    if main_h > 0 and main_w > 0:
        # Extract complete blocks
        main_region = image[:main_h * factor, :main_w * factor]
        blocks = _view_as_blocks(main_region, block_shape=(factor, factor))
        
        # Flatten blocks and compute modes
        flattened_blocks = blocks.reshape(main_h, main_w, -1)
        main_modes = np.apply_along_axis(fast_mode, -1, flattened_blocks)
        
        # Fill main region of output
        modes[:main_h, :main_w] = main_modes
    
    # Handle right edge (incomplete width blocks)
    if main_w < out_w:  # There are incomplete width blocks
        for i in range(main_h):
            start_h, end_h = i * factor, (i + 1) * factor
            start_w, end_w = main_w * factor, W

            block = image[start_h:end_h, start_w:end_w]
            modes[i, main_w] = fast_mode(block.flatten())
    
    # Handle bottom edge (incomplete height blocks)
    if main_h < out_h:  # There are incomplete height blocks
        for j in range(main_w):
            start_h, end_h = main_h * factor, H
            start_w, end_w = j * factor, (j + 1) * factor
            
            block = image[start_h:end_h, start_w:end_w]
            modes[main_h, j] = fast_mode(block.flatten())
    
    # Handle bottom-right corner (incomplete in both dimensions)
    if main_h < out_h and main_w < out_w:
        start_h, end_h = main_h * factor, H
        start_w, end_w = main_w * factor, W
        
        corner_block = image[start_h:end_h, start_w:end_w]
        modes[main_h, main_w] = fast_mode(corner_block.flatten())
    
    return modes 


def downsampled_images(image, scales=None):
    """
    Compute mode-based downsampling for multiple scales and store results.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array of shape (H, W) with non-negative integers
    scales : list[int] or None
        Optional list of block sizes. If None, uses power-of-2 scales up to min(H,W)//4
    
    Returns:
    --------
    dict
        Dictionary mapping scale -> downsampled 2D array
    """
    if scales is None:
        # Generate power-of-2 scales
        H, W = image.shape
        max_scale = max(1, min(H, W) // 2)
        powers = int(np.log2(max_scale)) + 1
        scales = [1] + [2 ** i for i in range(1, powers + 1) if 2 ** i <= max_scale]
        if len(scales) == 1:
            scales.append(2)
    
    downsampled_images = []
    for r in scales:
        downsampled_im = image.copy() if r == 1 else mode_downsample(image, r)
        if r==1 or downsampled_im.shape[0]*downsampled_im.shape[1] >= 16:
            downsampled_images.append(downsampled_im)

    return scales, downsampled_images

def compute_complexity_multiscale(image, scales, im_downsampled):
    """
    Compute normalized Zhang complexity K for a single 2D image using mode-based
    coarse-graining with fast bincount mode computation.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array of shape (H, W) with non-negative integers
    im_downsampled : list
        list of downsampled images
    
    Returns:
    --------
    float
        Normalized complexity value for the image
    """
    H, W = image.shape
    
    complexity = 0    
    for r, im_down in zip(scales, im_downsampled):
        # Compute histogram of mode values
        counts = collections.Counter(im_down.flatten())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        
        # Shannon entropy
        S_r = -np.sum(probs * np.log(probs))
        complexity += (r ** 2) * S_r # linear-r Riemann with delta_r = r/2 (scales grow by powers of two)
    
    # Normalize by total number of pixels
    K_norm = complexity / (H * W)
    return K_norm


def compute_complexity_levenshtein(image):
    """
    Computes the Levenshtein-based spatial complexity CL for a single 2D image.

    Parameters:
        image (2D numpy array): array of shape (R, C) with values 0–31 representing colors.

    Returns:
        float: the CL complexity score for the image
    """
    R, C = image.shape
    V_r = np.zeros(R - 1, dtype=np.uint8)
    V_c = np.zeros(C - 1, dtype=np.uint8)

    # Row-wise Levenshtein distances
    for i in range(R - 1):
        row1 = ''.join(map(chr, image[i]))
        row2 = ''.join(map(chr, image[i + 1]))
        V_r[i] = levenshtein_distance(row1, row2)

    # Column-wise Levenshtein distances
    for j in range(C - 1):
        col1 = ''.join(map(chr, image[:, j]))
        col2 = ''.join(map(chr, image[:, j + 1]))
        V_c[j] = levenshtein_distance(col1, col2)

    # Outer product and mean for CL
    A = np.outer(V_r, V_c)
    CL = A.mean()

    return CL

def compute_wavelet_energies(img, wavelet='haar', mid_scale = 0.1, large_scale = 0.4):
    """
    Compute high-, mid-, and low-frequency wavelet energies for a 2D integer-valued image.

    This function performs a multi-level 2D discrete wavelet transform (DWT) on an image of shape (H, W)
    and extracts three energy measures:
      1. High-frequency energy   → detail coefficients at level 1 (finest scale)
      2. Mid-frequency energy    → detail coefficients at the level whose center wavelength is closest
                                   to 10% of the smaller image dimension
      3. Low-frequency energy    → detail coefficients at the level whose center wavelength is closest
                                   to 40% of the smaller image dimension

    Level selection logic:
      • “level_high_f” is fixed to 1 (detail coefficients of the first DWT).
      • “level_mid_f” is chosen by finding L in [1..max_level] whose band-center (1.5·2^L) is nearest
        to 0.10 · min(H, W). If that coincides with level_high_f, it is bumped by +1.
      • “level_low_f” is chosen by finding L whose band-center is nearest to 0.40 · min(H, W),
        then adjusted upward if it collides with level_high_f or level_mid_f.

    After selecting levels, the function reconstructs all DWT subbands up through level_low_f and
    computes:
      • high_freq_energy = sum of squares of (LH, HL, HH) at level_high_f
      • mid_freq_energy  = sum of squares of (LH, HL, HH) at level_mid_f
      • low_freq_energy  = sum of squares of (LH, HL, HH) at level_low_f

    Parameters:
        img (np.ndarray of shape (H, W)):
            A 2D array of integer pixel labels (e.g., 0–31). Must be at least large enough to support
            the chosen wavelet decomposition for the desired levels.
        wavelet (str):
            Name of the wavelet to use for DWT (default: 'haar').

    Returns:
        tuple of floats:
            (low_freq_energy, mid_freq_energy, high_freq_energy)
            where each value is the sum of squared detail coefficients (LH, HL, HH) at the corresponding level.

    Raises:
        ValueError:
            If the image dimensions are too small to perform at least one level of DWT with the specified wavelet.
    """
    # Compute the maximum valid DWT level for this image + wavelet
    H, W = img.shape
    min_dim = min(H, W)
    max_pywt_level = pywt.dwtn_max_level((H, W), wavelet)
    if max_pywt_level < 1:
        warnings.warn(
            f"Image size {img.shape} is too small for wavelet '{wavelet}' decomposition. "
            "Returning zero energies."
        )
        return 0.0, 0.0, 0.0

    target_mid = mid_scale * min_dim
    target_low = large_scale * min_dim

    levels = np.arange(1, max_pywt_level + 1)
    scale_centers = 1.5 * (2 ** levels)

    l_ind_mid = np.argmin(np.abs(scale_centers - target_mid))
    l_ind_low = np.argmin(np.abs(scale_centers - target_low))

    level_high_f = 1
    level_mid_f = levels[l_ind_mid]
    level_low_f = levels[l_ind_low]

    # Ensure distinct levels for high, mid, and low
    if max(levels) > 1:
        if level_high_f == level_mid_f:
            level_mid_f = min(level_mid_f + 1, max(levels))
        if level_low_f == level_high_f or level_low_f == level_mid_f:
            level_low_f = min(level_low_f + 1, max(levels))

    coeffs = []
    current = img.copy()
    for i in range(level_low_f + 1):
        LL, (LH, HL, HH) = pywt.dwt2(current, wavelet)
        coeffs.append((LL, LH, HL, HH))
        current = LL

        if i == level_high_f:
            high_freq_energy = np.sum(LH**2 + HL**2 + HH**2)
        if i == level_mid_f:
            mid_freq_energy = np.sum(LH**2 + HL**2 + HH**2)
        if i == level_low_f:
            low_freq_energy = np.sum(LH**2 + HL**2 + HH**2)

    return low_freq_energy, mid_freq_energy, high_freq_energy


def compute_wavelet_energies_time(time_series, wavelet='db4'):
    ca, cd = pywt.dwt(time_series, wavelet=wavelet)
    low_freq_energy = np.sum(ca**2)
    high_freq_energy = np.sum(cd**2)
    return low_freq_energy, high_freq_energy

###############################################################################################
# The following functions come from:
# https://github.com/martiniani-lab/sweetsourcod/blob/master/examples/simple.py
# which is an example file from the sweetsourcod package developed by Stefano Martiniani et al.
###############################################################################################

@jit(nopython=True)
def mask_array(lattice, mask):
    return np.array([lattice[i] for i in mask])


def _get_entropy_rate(c, nsites, norm=1, alphabetsize=2, method='lz77'):
    """
    :param c: number of longest previous factors (lz77) or unique words (lz78)
    :param norm: normalization constant, usually the filesize per character of a random binary sequence of the same length
    :param method: lz77 or lz78
    :return: entropy rate h
    """
    if method == 'lz77':
        h = (c * np.log2(c) + 2 * c * np.log2(nsites / c)) / nsites
        h /= norm
    elif method == 'lz78':
        h = c * (np.log2(alphabetsize) + np.log2(c)) / nsites
        h /= norm
    else:
        raise NotImplementedError
    return h

def get_entropy_rate_lz77(x, extrapolate=True):
    # now with LZ77 we compute the number of longest previous factors c, and the entropy rate h
    # note that lz77 also outputs the sum of the logs of the factors which is approximately
    # equal to the compressed size of the file
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        c_bin, sumlog_bin = _lz77_complexity(random_binary_sequence)
        h_bound_bin = _get_entropy_rate(c_bin, nsites, norm=1, alphabetsize=np.unique(x).size, method='lz77')
        h_sumlog_bin = sumlog_bin
    else:
        h_bin = 1
        h_sumlog_bin = nsites
    c, h_sumlog = _lz77_complexity(x)
    h_bound = _get_entropy_rate(c, nsites, norm=h_bound_bin, alphabetsize=np.unique(x).size, method='lz77')
    h_sumlog /= h_sumlog_bin
    return h_bound, h_sumlog


def get_entropy_rate_lz78(x, extrapolate=True):
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        c_bin = _lz78_complexity(random_binary_sequence)
        h_bound_bin = _get_entropy_rate(c_bin, nsites, norm=1, alphabetsize=np.unique(x).size, method='lz78')
    else:
        h_bin = 1
    c = _lz78_complexity(x)
    h_bound = _get_entropy_rate(c, nsites, norm=h_bound_bin, alphabetsize=np.unique(x).size, method='lz78')
    return h_bound


def get_entropy_rate_bbox(x, extrapolate=True, algorithm='deflate', **kwargs):
    nsites = len(x)
    if extrapolate:
        random_binary_sequence = np.random.randint(0, 2, nsites, dtype='uint8')
        enc, _, _ = get_comp_size_bytes(random_binary_sequence, algorithm=algorithm, **kwargs)
        h_bin = 8 * enc / nsites
    else:
        h_bin = 1
    enc, _, _ = get_comp_size_bytes(x, algorithm=algorithm, **kwargs)
    h = 8 * enc / nsites
    return h / h_bin
