import matplotlib.pyplot as plt


def plot_signal(sig):
    """Simple plot of signal"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Signal")
    ax.plot(sig, color='C0')
    ax.set_xlabel("Sample [n]")
    ax.set_ylabel("Amplitude")

    plt.show()


def plot_psd(sig, freq):
    """Simple plot of psd"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Log of PSD")
    ax.plot(freq, sig, color='C0')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")

    plt.show()


def plot_lpc_errors(errors):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Normalized LPC error based on the number of coefficients")
    ax.plot(errors, color='C0')
    ax.set_xlabel("Coefficients")
    ax.set_ylabel("Error")

    plt.show()


def plot_psd_variants(fft, lpc, mel):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Comparison of power spectral density computed with different methods")
    ax.plot(fft[0], fft[1], label='fft')
    for lpc_psd, label in lpc[1]:
        ax.plot(lpc[0], lpc_psd, label=label)
    ax.plot(mel[0], mel[1], label='mel')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.legend()

    plt.show()
