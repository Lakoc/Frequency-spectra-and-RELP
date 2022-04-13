import numpy as np
from scipy.linalg import solve_toeplitz
import scipy.signal as sg
from features import mel_fbank_mx
import librosa


def split_padded(sig, win_size):
    """Pads signal with 0 and split equally to n windows"""
    n_windows = np.ceil(sig.shape[0] / win_size).astype(int)
    padding = (n_windows * win_size) - sig.shape[0]
    return np.array_split(np.concatenate((sig, np.zeros(padding))), n_windows)


def psd_fft(frame, w_size, n_coef):
    """Calculate power spectral density from the coefficients of the furrier transform"""
    windowed = np.hanning(w_size) * frame
    freq = np.fft.fft(windowed, n=n_coef)
    return (np.abs(freq) ** 2) / w_size


def lpc(frame, P):
    """Process linear prediction coding"""
    # Calculate correlation coefficients 0 -P
    R = np.array([np.sum(frame[i:] * frame[:len(frame) - i]) for i in range(P + 1)])

    # Create Toeplitz matrix
    c = R[0:P]  # first column
    b = -R[1:]

    # Solve toeplitz matrix with Levinson Durbin recursion
    A = solve_toeplitz(c, b)  # Levinson-Durbin recursion is called from the scipy toolkit

    # Calculate filter gain
    error_energy = R[0] + np.sum(A * R[1:])
    gain = np.sqrt(error_energy / frame.shape[0])

    return A, gain


def to_float16(val_tuple):
    return [item.astype(np.float16) for item in val_tuple]


def psd_lpc(A, G, n_coef):
    """Calculate power spectral density from the lpc coefficients."""
    h = sg.freqz(G, A, n_coef)[1]  # impulse response of the filter
    return np.abs(h) ** 2


def psd_mel(frame, w_size, nfft, n_banks, Fs, show_freq):
    """Calculate power spectral density in mel domain"""
    windowed = np.hanning(w_size) * frame
    psd = (np.abs(np.fft.fft(windowed, n=nfft)[:show_freq]) ** 2) / w_size
    mel_filters = mel_fbank_mx(nfft, Fs, NUMCHANS=n_banks)
    mel_psd = psd @ mel_filters

    freq = np.argmax(mel_filters, axis=0) / show_freq * (Fs / 2)
    return mel_psd, freq, mel_filters


def get_lpc_psd(A, G, freq_to_show):
    """Add 0th coeff to the filter and convert to log domain"""
    A = np.append(1.0, A)
    psd = 10 * np.log10(psd_lpc(A, G, freq_to_show))
    return psd


def get_residuals(frames, filters):
    """Get residuals from all frames by keeping filter state"""
    filter_init = np.zeros(filters[0][0].shape)
    residuals = []

    for index, frame in enumerate(frames):
        A = np.append(1.0, filters[index][0])  # appending with 1 for filtering
        residual, filter_state = sg.lfilter(A, [1], frame, zi=filter_init)
        filter_init = filter_state
        residuals.append(residual)
    return residuals


def decode_from_residuals(residuals, filters):
    """Create signal from residuals and LPC coefficients"""
    filter_init = np.zeros(filters[0][0].shape)
    sig_synthetized = []
    for index, residual in enumerate(residuals):
        A = np.append(1.0, filters[index][0])  # appending with 1 for filtering
        synthetized, filter_state = sg.lfilter([1.0], A, residual, zi=filter_init)
        filter_init = filter_state
        sig_synthetized.append(synthetized)
    return sig_synthetized


def nccf(signal, frame, start_index, end_index, min_lag, max_lag, prepended_vals):
    """Calculate normalized cross-correlation coefficients in range from min to max lag"""
    energy_1 = np.sum(frame ** 2)
    nccfs = np.zeros(max_lag + 1)
    for n in range(min_lag, max_lag + 1):
        # shift signal
        sig_shifted = signal[start_index - n + prepended_vals: end_index - n + prepended_vals]
        energy_2 = np.sum(sig_shifted ** 2)
        nccfs[n] = np.sum(frame * sig_shifted) / (np.sqrt(energy_1 * energy_2))
    return nccfs


def get_second_residuals(max_lag, min_lag, threshold, residuals, residuals_prepended, win_size):
    filter_init = np.zeros(max_lag)
    prev_lag = 0
    residuals_filtered = np.empty_like(residuals)
    lags = np.zeros(len(residuals), dtype=int)
    B = np.zeros(max_lag + 1)
    B[0] = 1.0
    # for each residual calculate all possible lags in specified range
    for index, frame in enumerate(residuals):
        nccfs = nccf(residuals_prepended, prepended_vals=max_lag + 1, end_index=(index + 1) * win_size,
                     min_lag=min_lag, max_lag=max_lag, start_index=index * win_size, frame=frame)

        # voiced segment
        if np.any(nccfs > threshold):
            lag = np.argmax(nccfs)

            # different lag in this segment so clear up filter memory
            if np.abs(prev_lag - lag) > 2:
                filter_init[:] = 0
            prev_lag = lag
            lags[index] = lag

            # init filter
            B[1:] = 0.0
            B[lag] = -1.0

            filtered, filter_state = sg.lfilter(B, [1.0], frame, zi=filter_init)
            filter_init = filter_state

        # non-voiced, do not filter pitch
        else:
            filter_init[:] = 0
            filtered = frame

        residuals_filtered[index] = filtered
    return residuals_filtered, lags


def get_second_residuals_Gaussian(max_lag, min_lag, threshold_voiced, threshold_unvoiced, residuals,
                                  residuals_prepended, win_size):
    filter_init = np.zeros(max_lag)
    prev_lag = 0
    residuals_voiced = []
    residuals_unvoiced = []
    unvoiced_indexes = []
    lags = np.zeros(len(residuals), dtype=int)
    B = np.zeros(max_lag + 1)
    B[0] = 1.0
    # for each residual calculate all possible lags in specified range
    for index, frame in enumerate(residuals):
        nccfs = nccf(residuals_prepended, prepended_vals=max_lag + 1, end_index=(index + 1) * win_size,
                     min_lag=min_lag, max_lag=max_lag, start_index=index * win_size, frame=frame)

        # voiced segment
        if np.any(nccfs > threshold_voiced):
            lag = np.argmax(nccfs)

            # different lag in this segment so clear up filter memory
            if np.abs(prev_lag - lag) > 2:
                filter_init[:] = 0
            prev_lag = lag
            lags[index] = lag

            # init filter
            B[1:] = 0.0
            B[lag] = -1.0

            filtered, filter_state = sg.lfilter(B, [1.0], frame, zi=filter_init)
            filter_init = filter_state
            residuals_voiced.append(filtered)
        elif np.all(nccfs < threshold_unvoiced):
            filter_init[:] = 0
            filtered = (np.mean(frame).astype(np.float32), np.std(frame).astype(np.float32))
            residuals_unvoiced.append(filtered)
            unvoiced_indexes.append(index)
        else:
            filter_init[:] = 0
            filtered = frame
            residuals_voiced.append(filtered)
    return residuals_voiced, residuals_unvoiced, np.array(unvoiced_indexes, dtype=np.uint32), lags


def decode_from_residuals2(residuals, lags, max_lag):
    """Create signal from residuals of residuals"""
    filter_init = np.zeros(max_lag)
    sig_synthetized = []
    B = np.zeros(max_lag + 1)
    B[0] = 1.0
    prev_lag = 0
    for index, residual in enumerate(residuals):
        lag = lags[index]
        if np.abs(prev_lag - lag) > 2:
            filter_init[:] = 0
        prev_lag = lag
        if lag == 0:
            sig_synthetized.append(residual)
            filter_init[:] = 0
        else:
            # init filter
            B[1:] = 0.0
            B[lag] = -1.0
            synthetized, filter_state = sg.lfilter([1.0], B, residual, zi=filter_init)
            filter_init = filter_state
            sig_synthetized.append(synthetized)
    return sig_synthetized


def sigma_clip(arr, a):
    """Process sigma clipping of values outside mean +- a*sigma"""
    mean = np.median(arr, axis=None)
    std = np.std(arr, axis=None)
    arr[arr < mean - a * std] = mean - a * std
    arr[arr > mean + a * std] = mean + a * std
    return arr


def quantize(arr, bits_to_quantize):
    """Quantize signal to the n bits, in logarithmic ration clipping outliers"""

    if len(arr) == 0:
        return arr, np.array(0), np.array(0), np.array(0)

    arr_min = np.min(arr, axis=None)
    # Convert to log domain, to better differentiate higher amplitudes,
    # log of negative number is nan, so add minimum to get rid of nans
    if arr_min < 0:
        arr += np.abs(arr_min) + np.finfo(float).eps
    arr = np.log(arr)

    # Clip outliers
    arr = sigma_clip(arr, 3)

    # Quantization step
    levels = 2 ** bits_to_quantize
    q = (np.max(arr) - np.min(arr)) / levels

    # Quantize and shift to the uint domain
    quantized = arr // q
    quantized_shift = np.abs(np.min(quantized))
    quantized = (quantized + quantized_shift).astype(np.uint8)

    return quantized, quantized_shift, q, arr_min


def dequantize(arr, quantized_shift, q, arr_min):
    """Create original arr from quantized one"""
    arr = arr - quantized_shift
    arr *= q
    arr = np.exp(arr)
    if arr_min < 0:
        arr -= np.abs(arr_min)
    return arr


def encode_first_order_residuals(wav_file, Fs, win_size, P, L_bounds):
    sig, Fs = librosa.load(wav_file, sr=Fs)
    sig = sig - np.mean(sig)
    frames = split_padded(sig, win_size)
    lpc_filters = [to_float16(lpc(frame, P)) for frame in frames]
    residuals = get_residuals(frames, lpc_filters)
    residuals_prepended = np.pad(np.hstack(residuals), (L_bounds[1] + 1, 0), constant_values=0)
    return residuals, residuals_prepended, lpc_filters


def encoder_quantize(residuals_filtered, residual_bits, lags):
    voiced_quantized, q_shift, q, min_val = quantize(residuals_filtered, residual_bits)
    lags = lags.astype(np.uint8)
    q_shift = q_shift.astype(np.uint8)
    q = q.astype(np.float32)
    min_val = min_val.astype(np.float32)
    return voiced_quantized, q_shift, q, min_val, lags


def encode_wav(wav_file, Fs, win_size, P, L_bounds, voiced_thr, residual_bits):
    """Full encoder unit
    1. load and frame signal
    2. calculate lpc
    3. get residuals
    4. get second order residuals filtered by pitch
    5. quantize second order residuals and cast other coefficients to smaller types"""
    residuals, residuals_prepended, lpc_filters = encode_first_order_residuals(wav_file, Fs, win_size, P, L_bounds)
    so_residuals, lags = get_second_residuals(L_bounds[1], L_bounds[0], threshold=voiced_thr,
                                              residuals=residuals,
                                              residuals_prepended=residuals_prepended, win_size=win_size)
    return *encoder_quantize(so_residuals, residual_bits, lags), lpc_filters


def encode_wav_gaussian(wav_file, Fs, win_size, P, L_bounds, residual_bits, voiced_thr, unvoiced_thr=None):
    """Full encoder unit with unvoiced frames modeled by Gaussian noise
    1. load and frame signal
    2. calculate lpc
    3. get residuals
    4. get second order residuals filtered by pitch and modeled with Gaussian noise if weak correlation found
    5. quantize second order residuals and cast other coefficients to smaller types"""
    residuals, residuals_prepended, lpc_filters = encode_first_order_residuals(wav_file, Fs, win_size, P, L_bounds)
    voiced_residuals, unvoiced_residuals, unvoiced_indexes, lags = \
        get_second_residuals_Gaussian(L_bounds[1],
                                      L_bounds[0],
                                      threshold_voiced=voiced_thr,
                                      threshold_unvoiced=unvoiced_thr or (
                                              1 - voiced_thr),
                                      residuals=residuals,
                                      residuals_prepended=residuals_prepended,
                                      win_size=win_size)
    return *encoder_quantize(voiced_residuals, residual_bits, lags), unvoiced_residuals, unvoiced_indexes, lpc_filters


def decode_wav(voiced_quantized, q_shift, q, min_val, lags, lpc_filters, L_max):
    """Decoder unit
    1. dequantize second order residuals
    2. decode second order residuals to obtain first order residuals
    3. decode first order residuals"""
    q_shift = q_shift.astype(np.float32)
    res_orig = dequantize(voiced_quantized, q_shift, q, min_val)
    lags = lags.astype(np.int32)
    decoded_sig = np.hstack(
        decode_from_residuals(decode_from_residuals2(res_orig, lags, L_max), lpc_filters))
    return decoded_sig


def decode_wav_gaussian(voiced_quantized, unvoiced, unvoiced_indexes, q_shift, q, min_val, lags, lpc_filters, L_max,
                        residual_size):
    """Decoder unit with unvoiced residuals modeled by Gaussian noise
     1. dequantize second order residuals
     2. decode second order residuals to obtain first order residuals
     3. decode first order residuals"""
    q_shift = q_shift.astype(np.float32)
    voiced = dequantize(voiced_quantized, q_shift, q, min_val)
    res_orig = np.empty((voiced.shape[0] + unvoiced_indexes.shape[0], residual_size))

    unvoiced_index = 0
    voiced_index = 0
    for index in range(res_orig.shape[0]):
        if index in unvoiced_indexes:
            residual = np.random.normal(unvoiced[unvoiced_index][0], unvoiced[unvoiced_index][1], res_orig.shape[1])
            unvoiced_index += 1
        else:
            residual = voiced[voiced_index]
            voiced_index += 1
        res_orig[index] = residual
    lags = lags.astype(np.int32)
    decoded_sig = np.hstack(
        decode_from_residuals(decode_from_residuals2(res_orig, lags, L_max), lpc_filters))
    return decoded_sig


def print_compression_stats(original_size, voiced_quantized, lpc_filters, q_shift, q, lags, residual_bits, min_val,
                            L_max, unvoiced, unvoiced_indexes, compressed_indexes_bytes=0):
    """Calculate compression stats and print them to the stdout"""
    bits_per_byte = 8
    print(f'Original wav size = {original_size / 1000:.2f} kB')
    voiced_size = 0
    unvoiced_size = 0
    unvoiced_part = ''
    voiced_part = ''
    if len(voiced_quantized) > 0:
        voiced_size = voiced_quantized.shape[0] * (
                lpc_filters[0][1].nbytes + lpc_filters[0][0].nbytes + lags[0].nbytes + (
                int(voiced_quantized.shape[1] * residual_bits) / bits_per_byte)) + \
                      q_shift.nbytes + q.nbytes + min_val.nbytes + compressed_indexes_bytes
        voiced_part = f' {voiced_quantized.shape[0]} [Number_of_voiced_frames] * ({lpc_filters[0][1].nbytes} ' \
                      f'[Gain, float_16] + {lpc_filters[0][0].nbytes} [LPC_filters, 7 float_16]+ {lags[0].nbytes} ' \
                      f'[Lag, uint_8] + {int((voiced_quantized.shape[1] * residual_bits) / bits_per_byte)} ' \
                      f'[Residual_quantized, {voiced_quantized.shape[1]} uint_{residual_bits}]) + ' \
                      f'{q_shift.nbytes} [q_shift, uint_8] + {q.nbytes} [q, float_32] + {min_val.nbytes} ' \
                      f'[min_val, float_32]'
        if compressed_indexes_bytes > 0:
            voiced_part += f' + {compressed_indexes_bytes} [compressed_indexes, {compressed_indexes_bytes} uint_8]'
    if unvoiced is not None:
        unvoiced_size = len(unvoiced) * (unvoiced[0][0].nbytes + unvoiced[0][1].nbytes) + unvoiced_indexes.nbytes

        unvoiced_part += f' {len(unvoiced)} [Number_of_unvoiced_frames] * ({unvoiced[0][0].nbytes}' \
                         f' [Gaussian mean, float32] + {unvoiced[0][1].nbytes} [Gaussian std, float32])' \
                         f' + {unvoiced_indexes.nbytes} [unvoiced_indexes, {unvoiced_indexes.shape[0]} uint_32] +'
    coded_size = int(
        voiced_size + unvoiced_size + L_max.nbytes)
    print(
        f'Coded signal size ={voiced_part}{unvoiced_part} {L_max.nbytes} [L_max, uint_8] '
        f'= {coded_size * 8} bits = {coded_size / 1000:.2f} kB')
    print(f'Compression ratio = {original_size / coded_size:.2f}')


def compress_less_levels(quantized_residuals, reduce_to_n_bits):
    """Compress array to n-bits by iteratively selecting the least common element
     and replacing values in arr by neighboring element in histogram."""
    frames = np.hstack(quantized_residuals).astype(int)
    vals, counts = np.unique(frames, return_counts=True)

    # Calculate how many indexes should be thrown away
    indexes_to_throw = counts.shape[0] - (2 ** reduce_to_n_bits)
    for throw in range(indexes_to_throw):
        vals, counts = np.unique(frames, return_counts=True)
        smallest_val = np.argsort(counts)[0]
        val_to_be_replaced = vals[smallest_val]
        smaller = vals[smallest_val - 1] if (smallest_val - 1 >= 0) else -1000  # constant which is always further
        higher = vals[smallest_val + 1] if (
                smallest_val + 1 < vals.shape[0]) else -1000  # constant which is always further
        val_to_replace = higher if np.abs(val_to_be_replaced - higher) < np.abs(
            val_to_be_replaced - smaller) else smaller  # find closer neighbor
        frames[frames == val_to_be_replaced] = val_to_replace

    indexes = np.unique(frames, return_counts=True)[0]
    # Replace values with indexes to aux array
    for index in range(indexes.shape[0]):
        frames[frames == indexes[index]] = index
    return frames.reshape(quantized_residuals.shape).astype(np.uint8), indexes.astype(np.uint8)
