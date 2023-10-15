import numpy
import scipy
import wave
import matplotlib.pyplot as pyplot


def part1_2():
    """
    1. The QZ3Viola.wav file is a recording of a viola playing a single note. Use the PSD
    to find an estimate of the fundamental frequency of this signal. Indicate
    parameters used to compute the PSD (window size, tapering, overlap, number
    of fft points) and show the plot from which you derived your answer. Zoom
    into the critical part of the plot if needed to clearly show how you determined
    this number.
    """
    # read the wav file
    with wave.open('QZ3Viola.wav', 'rb') as f:
        # get the sampling frequency
        sampling_frequency = f.getframerate()
        # get the number of samples
        num_samples = f.getnframes()
        # read the samples
        samples = numpy.frombuffer(f.readframes(num_samples), dtype=numpy.int16)

    # compute the PSD
    f, pxx = scipy.signal.welch(samples, fs=sampling_frequency, window='hamming', nperseg=8192, noverlap=512, nfft=8192)
    # plot the PSD
    pyplot.figure()
    pyplot.title('1) PSD')
    pyplot.plot(f, pxx)
    pyplot.xlabel('Frequency [Hz]')
    pyplot.ylabel('Magnitude')
    pyplot.xlim(100, 600)
    pyplot.show()

    # print the window size, tapering, overlap, and number of fft points
    print('1) Parameters used to compute the PSD:')
    print('Window size: 8192')
    print('Tapering: Hamming')
    print('Overlap: 512')
    print('Number of fft points: 8192')

    """
    2. High-pass filter the viola signal with a cutoff of 1200 Hz (any type, any order)
    and plot the PSD of the filtered signal and indicate filter used. Did the
    fundamental frequency change as a result of the filtering (may want to look at
    the time waveform on this)?
    """
    # create the high-pass filter
    sos = scipy.signal.butter(10, 1200, 'highpass', fs=sampling_frequency, output='sos')
    # filter the signal
    filtered_samples = scipy.signal.sosfilt(sos, samples)
    # compute the PSD
    f, pxx = scipy.signal.welch(filtered_samples, fs=sampling_frequency, window='hamming', nperseg=8192, noverlap=512,
                                nfft=8192)
    # plot the PSD
    pyplot.figure()
    pyplot.title('2) PSD with High-pass Filter')
    pyplot.plot(f, pxx)
    pyplot.xlabel('Frequency [Hz]')
    pyplot.ylabel('Magnitude')
    # pyplot.ylim(0, 1e6)
    pyplot.xlim(21800, 21900)
    pyplot.show()

    # print the filter used
    # print('2) Filter used: Butterworth high-pass filter with a cutoff of 1200 Hz')


def part3():
    """
    3. The QZ3NoteSequence.wav file is a record of 4 notes played on a virtual piano.
    Use the spectrogram to determine the fundamental frequency of each note and
    the duration in seconds of each note. Show the computed spectrogram used
    and indicate the parameters used for computing the spectrogram (window
    size, taper, overlap, number of FFT points).
    """
    # read the wav file
    with wave.open('QZ3NoteSequence.wav', 'rb') as f:
        # get the sampling frequency
        sampling_frequency = f.getframerate()
        # get the number of samples
        num_samples = f.getnframes()
        # read the samples
        samples = numpy.frombuffer(f.readframes(num_samples), dtype=numpy.int16)

    # compute the spectrogram
    f, t, sxx = scipy.signal.spectrogram(samples, fs=sampling_frequency, window='hamming', nperseg=8192, noverlap=512,
                                         nfft=8192, mode='magnitude')
    # plot the spectrogram
    pyplot.figure()
    pyplot.title('3) Spectrogram')
    pyplot.pcolormesh(t, f, 20 * numpy.log10(sxx), shading='gouraud')
    pyplot.ylabel('Frequency [Hz]')
    pyplot.xlabel('Time [sec]')
    cbar = pyplot.colorbar()
    cbar.ax.set_ylabel('Magnitude [dB]')
    # make plot wider
    pyplot.gcf().set_size_inches(10, 5)
    pyplot.ylim(0, 1000)
    pyplot.xlim(0.7, 11)
    # increase tick marks on x-axis
    pyplot.xticks(numpy.arange(0.7, 11, 0.5))
    pyplot.show()

    # print the window size, tapering, overlap, and number of fft points
    print('3) Parameters used to compute the spectrogram:')
    print('Window size: 8192')
    print('Tapering: Hamming')
    print('Overlap: 512')
    print('Number of fft points: 8192')

    # find the indices of the max value in each column of the spectrogram
    max_indices = numpy.argmax(sxx, axis=0)
    # find the frequencies corresponding to the max values
    max_frequencies = f[max_indices]
    print(max_frequencies)


def part4():
    """
    The QZ3MultipathRoom.wav is simulated sound file of a pink noise burst where
    one dominant scatterer pair is causing multiple reflections received by the
    recording microphone. Use the autocorrelation to determine the time delay
    between the direct path and the first reflection received at the microphone of
    this recording. Include a plot of the autocorrelation use to make this estimate.
    """
    # read the wav file
    with wave.open('QZ3MultipathRoom.wav', 'rb') as f:
        # get the sampling frequency
        sampling_frequency = f.getframerate()
        # get the number of samples
        num_samples = f.getnframes()
        # read the samples
        samples = numpy.frombuffer(f.readframes(num_samples), dtype=numpy.int16)

    # normalize the samples
    normalized_samples = samples / numpy.std(samples)

    # compute the autocorrelation
    autocorrelation = scipy.signal.correlate(normalized_samples, normalized_samples, mode='same')
    # convert the time axis from samples to milliseconds
    time_axis = numpy.arange(len(autocorrelation)) / sampling_frequency * 1000

    # normalize the autocorrelation
    autocorrelation /= numpy.max(autocorrelation)

    # autocorrelation /= len(autocorrelation)

    # plot the autocorrelation
    pyplot.figure()
    pyplot.title('4) Autocorrelation')
    pyplot.plot(time_axis, autocorrelation)
    pyplot.xlabel('Time [milliseconds]')
    # increment the x axis by 50 milliseconds
    pyplot.xticks(numpy.arange(550, 600, 2))
    pyplot.xlim(550, 600)
    # make plot wider
    pyplot.gcf().set_size_inches(12, 5)
    pyplot.ylabel('Autocorrelation coefficient')
    pyplot.show()

    # find the index of the max value in the autocorrelation
    max_index = numpy.argmax(autocorrelation)
    # find the time difference between the direct path and the first reflection
    time_difference = max_index / sampling_frequency
    print(f'4) Time difference between the direct path and the first reflection: {time_difference} seconds')


if __name__ == "__main__":
    # part1_2()
    # part3()
    part4()
