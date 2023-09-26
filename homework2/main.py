import numpy
from scipy import signal
import matplotlib.pyplot as pyplot

if __name__ == "__main__":
    """
    2.1)
    
    Use the FFT to filter a 32 point square pulse s[n]:
    s[n]=1 for 11<=n<=21 else 0

    (index n starts a 0)
    where the FIR filter has impulse response h[n]:
    h[n] = (-0.75)^n for 30>n>=0 else 0 for 32>n>=30

    a) Use the fft command to take signals into frequency domain without
    padding with zeros. Multiply signal and transfer function, then take
    inverse fft to obtain time domain signal. Plot the filtered signals and
    explain what you observe.
    b) Repeat part (a) except use the fft command option to pad with zeros and
    double the signal lengths.
    """
    # Define the input signal
    N = 32
    n = numpy.arange(N)
    s = numpy.where((n >= 11) & (n <= 21), 1, 0)
    h = numpy.where(n < 30, (-0.75) ** n, 0)

    # a)
    # Take the FFT of the signals
    S = numpy.fft.fft(s)
    H = numpy.fft.fft(h)

    # Perform the filtering by multiplying the FFTs
    filtered_signal = numpy.fft.ifft(S * H)

    # Show the original and filtered signals
    pyplot.figure(figsize=(12, 6))

    pyplot.subplot(2, 1, 1)
    pyplot.plot(s, label='Original')
    pyplot.plot(h, label='Filter')
    pyplot.plot(numpy.abs(filtered_signal), label='Filtered')
    pyplot.title('2.1a) Without zero-padding')
    pyplot.legend()

    # b)
    # Zero-pad the signals
    sp = numpy.pad(s, (0, 32))
    hp = numpy.pad(h, (0, 32))

    # Take the FFT of the padded signals
    Sp = numpy.fft.fft(sp)
    Hp = numpy.fft.fft(hp)

    # Perform the filtering by multiplying the FFTs
    filtered_signal_padded = numpy.fft.ifft(Sp * Hp)

    pyplot.subplot(2, 1, 2)
    pyplot.plot(sp, label='Original')
    pyplot.plot(hp, label='Filter')
    pyplot.plot(numpy.abs(filtered_signal_padded), label='Filtered')
    pyplot.title('2.1b) With zero-padding')
    pyplot.legend()

    pyplot.tight_layout()
    pyplot.show()

    """
    2.2)
    Create a signal consisting of 2 sine waves with amplitude 1 and sampled at 8000 Hz. Set one with frequency 400 Hz 
    and the other with 404 Hz.
    
    a) Take the DFT using window length of 0.05, 0.25 and 0.5 seconds. Describe what you see and generalize about the 
    impact of signal length on frequency resolution.

    b) Repeat part (a) using zero padding so that each signal length is effectively 1 second. Describe what you see and 
    generalize about the impact zero padding on frequency resolution.
    """
    n = numpy.linspace(0, 0.5, num=int(0.5 * 8000), endpoint=False)
    signal_gen = numpy.sin(2 * numpy.pi * 400 * n) + numpy.sin(2 * numpy.pi * 404 * n)
    window_lengths = [0.05, 0.25, 0.5]

    # Part a
    fig, axs = pyplot.subplots(len(window_lengths), 1, figsize=(12, 6))

    for i, window_sec in enumerate(window_lengths):
        window_samples = int(window_sec * 8000)

        f, t, Sxx = signal.spectrogram(signal_gen[:window_samples], fs=8000, nperseg=window_samples, noverlap=0,
                                       window='boxcar', mode='magnitude')
        axs[i].set_title('2.2a) Spectrogram (window = {window} s)'.format(window=window_sec))
        axs[i].plot(f, Sxx.reshape(-1))
        axs[i].axvline(x=400, color='r', linestyle='--')
        axs[i].axvline(x=404, color='r', linestyle='--')
        axs[i].set_xlim(200, 600)

    pyplot.tight_layout()
    pyplot.show()

    # Part b
    fig, axs = pyplot.subplots(len(window_lengths), 1, figsize=(12, 6))

    for i, window_sec in enumerate(window_lengths):
        window_samples = int(window_sec * 8000)

        signal_padded = numpy.pad(signal_gen[:window_samples], (0, 1 * 8000 - window_samples))
        f, t, Sxx = signal.spectrogram(signal_padded, fs=8000, nperseg=8000, noverlap=0, window='boxcar',
                                       mode='magnitude')
        axs[i].set_title('2.2b) Spectrogram with zero padding (window = {window} s)'.format(window=window_sec))
        axs[i].plot(f, Sxx.reshape(-1))
        axs[i].axvline(x=400, color='r', linestyle='--')
        axs[i].axvline(x=404, color='r', linestyle='--')
        axs[i].set_xlim(200, 600)

    pyplot.tight_layout()
    pyplot.show()
