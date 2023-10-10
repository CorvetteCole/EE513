import numpy
import scipy
import matplotlib.pyplot as pyplot

from project1.tone_gen import ToneGenerator


def plot_spectrogram(tone, sampling_frequency, nperseg, nfft, noverlap, window, mode, title,
                     ylim=(256, 1024)):
    f, t, sxx = scipy.signal.spectrogram(tone, fs=sampling_frequency, nperseg=nperseg, nfft=nfft, noverlap=noverlap,
                                         window=window, mode=mode)
    pyplot.figure()
    pyplot.title(title)
    pyplot.pcolormesh(t, f, 20 * numpy.log10(sxx), shading='gouraud')
    pyplot.ylabel('Frequency [Hz]')
    pyplot.xlabel('Time [sec]')
    cbar = pyplot.colorbar()
    cbar.ax.set_ylabel('Magnitude [dB]')
    pyplot.ylim(ylim)
    pyplot.show()


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
    toneGenerator = ToneGenerator(256, sampling_frequency=4e3)
    tones = [toneGenerator.generate(i, 0.25) for i in range(0, 13)]
    # for tone in tones:
    #     toneGenerator.play(tone)
    scale = numpy.concatenate(tones)

    # Part a
    plot_spectrogram(scale, 4e3, 256, 256, 0, 'hamming', 'magnitude',
                     '3.1a) Spectrogram')
    # Part b
    plot_spectrogram(scale, 4e3, 256, 1024, 0, 'hamming', 'magnitude',
                     '3.1b) Spectrogram')
    # Part c
    plot_spectrogram(scale, 4e3, 1024, 2048, 0, 'hamming', 'magnitude',
                     '3.1c) Spectrogram')

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


    def plot_ac_psd(tone, mxlag, sampling_rate, name):
        tone_ac = scipy.signal.correlate(tone, tone, mode='full')
        tone_ac = tone_ac[:int(mxlag * sampling_rate)]
        tone_ac /= len(tone_ac)

        tone_freq, tone_psd = scipy.signal.welch(tone, fs=sampling_rate, nperseg=16384)

        pyplot.figure(figsize=(10, 6))

        # AC plot
        pyplot.subplot(2, 1, 1)
        pyplot.plot(tone_ac)
        pyplot.xlabel('milliseconds (ms)')
        pyplot.ylabel('Auto-correlation')
        pyplot.title(f'Auto-correlation of {name}')

        # PSD plot
        pyplot.subplot(2, 1, 2)
        pyplot.plot(tone_freq, tone_psd)
        pyplot.xlabel('Frequency (Hz)')
        pyplot.xlim(0, 500)
        pyplot.ylim(0, 16e-4)
        pyplot.ylabel('Magnitude')
        pyplot.title(f'Power Spectral Density of {name}')

        pyplot.tight_layout()
        pyplot.show()
        pass


    bnum = [0.04957526213389, -0.06305581334498, 0.01483220320740]
    aden = [1.00000000000000, -1.80116083982126, 0.80257737639225]

    sampling_rate = 44100  # Sampling rate of 44.1 kHz
    time = 3  # 3 seconds of noise
    num_samples = sampling_rate * time
    mxlag = 0.02

    # Generating white noise
    white_noise = numpy.random.normal(0, 1, num_samples)

    # Applying the filter to the white noise to generate pink noise
    pink_noise = scipy.signal.lfilter(bnum, aden, white_noise)

    plot_ac_psd(white_noise, mxlag, sampling_rate, 'White Noise')
    plot_ac_psd(pink_noise, mxlag, sampling_rate, 'Pink Noise')


    # part b
    # Sinusoid frequency
    frequency = 220

    # Sinusoid power
    pink_noise_power = numpy.std(pink_noise)
    sinusoid_amplitude = numpy.sqrt(2) * pink_noise_power

    t = numpy.arange(num_samples) / sampling_rate
    sinusoid = sinusoid_amplitude * numpy.sin(2 * numpy.pi * frequency * t)

    # Adding sinusoid to pink noise
    combined_signal = pink_noise + sinusoid

    plot_ac_psd(combined_signal, mxlag, sampling_rate, 'Pink Noise with Sinusoid')
