import numpy as np
from scipy.linalg import solve_toeplitz
import scipy.signal as sg
from features import mel_fbank_mx


def split_padded(sig, win_size):
    """Pads signal with 0 and split equally to n windows"""
    n_windows = np.ceil(sig.shape[0] / win_size).astype(int)
    padding = (-sig.shape[0]) % n_windows
    return np.split(np.concatenate((sig, np.zeros(padding))), n_windows)


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

    # Solve with Levinson Durbin recursion
    A = solve_toeplitz(c, b)

    # Calculate filter gain
    error_energy = R[0] + np.sum(A * R[1:])
    gain = np.sqrt(error_energy / frame.shape[0])

    return A, gain


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
    return mel_psd, freq


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
    filter_init = np.zeros(max_lag - 1)
    prev_lag = 0
    residuals_filtered = np.empty_like(residuals)
    lags = np.zeros(len(residuals), dtype=int)
    B = np.zeros(max_lag)
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


def decode_from_residuals2(residuals, lags, max_lag):
    """Create signal from residuals of residuals"""
    filter_init = np.zeros(max_lag - 1)
    sig_synthetized = []
    B = np.zeros(max_lag)
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
