import numpy
from scipy import signal
import matplotlib.pyplot as pyplot

from project1.tone_gen import ToneGenerator


def plot_spectrogram(figure: pyplot.figure, tone, sampling_frequency, nperseg, nfft, noverlap, window, mode, title,
                     ylim=(256, 512)):
    f, t, sxx = signal.spectrogram(tone, fs=sampling_frequency, nperseg=nperseg, nfft=nfft, noverlap=noverlap,
                                   window=window, mode=mode)
    figure.set_title(title)
    figure.pcolormesh(t, f, sxx, shading='gouraud')
    figure.set_ylabel('Frequency [Hz]')
    figure.set_xlabel('Time [sec]')
    figure.set_ylim(ylim)


if __name__ == "__main__":
    """
    3.1) 
    
    Use the scale program created in a previous assignment (may need to
    fix it up) to generate a scale for 2 octaves starting at 256 Hz with tones
    of amplitude 1 and duration 0.25 seconds. Use a sampling rate of
    4kHz.
    
    a) Compute and plot the spectrogram magnitude in dB, labeling all axes
    correctly. Use the parameters you think best for estimating the
    spectrogram for this signal. Comment on the frequencies of the tones
    generated with your program and those identified through the
    spectrogram.
    
    b) Quadruple the number of FFT points from that used in (A). Compute
    and plot the spectrogram, and explain difference observe from part
    (a).
    
    c) Increase the window length by a factor of 4 and set the number of FFT
    points to twice the amount of the window length. Compute and plot
    the spectrogram and explain the observed changes.
    """
    toneGenerator = ToneGenerator(256, 12, 4e3)
    tones = [toneGenerator.generate(i, 0.25) for i in range(0, 13)]
    # for tone in tones:
    #     toneGenerator.play(tone)
    scale = numpy.concatenate(tones)

    fig, axs = pyplot.subplots(3, 1, figsize=(6, 12))
    # Part a
    plot_spectrogram(axs[0], scale, 4e3, 256, 256, 0, 'boxcar', 'magnitude',
                     '3.1a) Spectrogram (window = 256 samples)')
    # Part b
    plot_spectrogram(axs[1], scale, 4e3, 256, 1024, 0, 'boxcar', 'magnitude',
                     '3.1b) Spectrogram (window = 256 samples, nfft = 1024)')
    # Part c
    plot_spectrogram(axs[2], scale, 4e3, 1024, 2048, 0, 'boxcar', 'magnitude',
                     '3.1c) Spectrogram (window = 1024 samples, nfft = 2048)')

    pyplot.tight_layout()
    pyplot.show()

    """
    3.2)
    
    Generate 3 seconds of white noise at a sampling rate of 44.1 kHz. Use
    the following coefficients in an IIR filter to filter the white noise and
    generate pink noise (bnum are numerator and aden are denominator
    coefficients).
    
    bnum = [ 0.04957526213389, -0.06305581334498, 0.01483220320740 ]
    aden = [ 1.00000000000000, -1.80116083982126, 0.80257737639225]
    
    a) Estimate the AC and PSD of the Pink noise and present their plots.
    Comment on the observed differences from white noise.
    
    b) Add a sinusoid of frequency 220 Hz to the pink noise signal with
    power equal to that of the pink noise (i.e. pink noise power computed
    from: p = std(pink noise signal) and sinusoidal power can be set equal
    to that by: sig = sqrt(2)*p*sin()) repeat part (a). Show the plots and
    explain the differences observed with the plots from part (a). Does
    this make sense since the sine function is not random?
    """
    # Part a
    white_noise = numpy.random.normal(0, 1, int(3 * 44.1e3))
    bnum = [0.04957526213389, -0.06305581334498, 0.01483220320740]
    aden = [1.00000000000000, -1.80116083982126, 0.80257737639225]
    pink_noise = signal.lfilter(bnum, aden, white_noise)

    # calculate auto-correlation and power spectral density of the pink noise and white noise
    white_noise_ac = signal.correlate(white_noise, white_noise)
    white_noise_psd = numpy.abs(numpy.fft.fft(white_noise_ac))
    pink_noise_ac = signal.correlate(pink_noise, pink_noise)
    pink_noise_psd = numpy.abs(numpy.fft.fft(pink_noise_ac))

    fig, axs = pyplot.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('3.2a) White Noise vs. Pink Noise')
    axs[0, 0].plot(white_noise_ac)
    axs[0, 0].set_title('White Noise Auto-Correlation')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Correlation Coefficient')
    axs[0, 1].plot(white_noise_psd)
    axs[0, 1].set_title('White Noise Power Spectral Density')
    axs[0, 1].set_xlabel('Frequency')
    axs[0, 1].set_ylabel('Magnitude')
    axs[1, 0].plot(pink_noise_ac)
    axs[1, 0].set_title('Pink Noise Auto-Correlation')
    axs[1, 0].set_xlabel('Sample')
    axs[1, 0].set_ylabel('Correlation Coefficient')
    axs[1, 1].plot(pink_noise_psd)
    axs[1, 1].set_title('Pink Noise Power Spectral Density')
    axs[1, 1].set_xlabel('Frequency')
    axs[1, 1].set_ylabel('Magnitude')
    pyplot.tight_layout()
    pyplot.show()

    # Part b
    sine = numpy.sqrt(2) * numpy.std(pink_noise) * numpy.sin(2 * numpy.pi * 220 * numpy.linspace(0, 3, int(3 * 44.1e3)))
    pink_noise_with_sine = pink_noise + sine

    # calculate auto-correlation and power spectral density of the pink noise
    pink_noise_with_sine_ac = signal.correlate(pink_noise_with_sine, pink_noise_with_sine)
    pink_noise_with_sine_psd = numpy.abs(numpy.fft.fft(pink_noise_with_sine_ac))

    fig, axs = pyplot.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('3.2b) Pink Noise vs. Pink Noise with Sine')
    axs[0, 0].plot(pink_noise_ac)
    axs[0, 0].set_title('Pink Noise Auto-Correlation')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Correlation Coefficient')
    axs[0, 1].plot(pink_noise_psd)
    axs[0, 1].set_title('Pink Noise Power Spectral Density')
    axs[0, 1].set_xlabel('Frequency')
    axs[0, 1].set_ylabel('Magnitude')
    axs[1, 0].plot(pink_noise_with_sine_ac)
    axs[1, 0].set_title('Pink Noise with Sine Auto-Correlation')
    axs[1, 0].set_xlabel('Sample')
    axs[1, 0].set_ylabel('Correlation Coefficient')
    axs[1, 1].plot(pink_noise_with_sine_psd)
    axs[1, 1].set_title('Pink Noise with Sine Power Spectral Density')
    axs[1, 1].set_xlabel('Frequency')
    axs[1, 1].set_ylabel('Magnitude')
    pyplot.tight_layout()
    pyplot.show()

