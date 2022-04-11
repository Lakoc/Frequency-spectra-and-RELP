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


def plot_original_vs_residual(sig, residual):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Comparison of residual and original system of selected frame")
    ax.plot(sig, label='original')
    ax.plot(residual, label='residual')
    ax.set_xlabel("Sample [n]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()


def plot_residual_vs_residual2(residual, residual2):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Comparison of lpc residual and second residual")
    ax.plot(residual, label='residual')
    ax.plot(residual2, label='residual with pitch filtered')
    ax.set_xlabel("Sample [n]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()
