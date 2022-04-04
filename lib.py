import numpy as np
from scipy.linalg import solve_toeplitz
import scipy.signal as sg
from features import mel_fbank_mx


def split_padded(sig, n_windows):
    """Pads signal with 0 and split equally to n windows"""
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
    """
    Calculate power spectral density from the lpc coefficients.
    """
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
    A = np.append(1.0, A[:-1])
    psd = 10 * np.log10(psd_lpc(A, G, freq_to_show))
    return psd
