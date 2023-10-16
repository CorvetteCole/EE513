import numpy
import scipy
import matplotlib.pyplot as pyplot


def total_harmonic_distortion(signal: numpy.ndarray, sampling_frequency: float, num_harmonics=5):
    """
    Calculates the total harmonic distortion (THD) in dBc of the real-valued sinusoidal signal. The total harmonic
    distortion is determined from the fundamental frequency and the first five harmonics using a modified periodogram of
    the same length as the input signal. The modified periodogram uses a Kaiser window with β = 38.

    :param signal: Real-valued sinusoidal input signal
    :param sampling_frequency: Sample rate of the input signal in Hz
    :param num_harmonics: Number of harmonics to use in the THD calculation
    :return: The total harmonic distortion in dBc
    """
    # Calculate the periodogram of the input signal using a Kaiser window with beta = 38
    f, pxx = scipy.signal.periodogram(signal, fs=sampling_frequency, window='kaiser', scaling='spectrum',
                                      nfft=len(signal), return_onesided=True)

    # Find the fundamental frequency and its index in the periodogram
    fundamental_index = numpy.argmax(pxx)
    # fundamental_frequency = f[fundamental_index] # TODO remove

    # Calculate the THD using the fundamental frequency and the first num_harmonics harmonics
    harmonic_indices = [fundamental_index * (i + 1) for i in range(num_harmonics)]
    harmonic_powers = pxx[harmonic_indices]
    return 10 * numpy.log10(numpy.sum(harmonic_powers) / pxx[fundamental_index])


def center_clip(signal: numpy.ndarray, threshold: float, reduce_amplitude=False):
    """
    Returns a center clipped version of the input signal based on the threshold.

    :param signal: The signal to center clip
    :param threshold: The threshold to center clip the signal at
    :param reduce_amplitude: Whether to reduce the amplitude of the signal by the threshold
    :return: The center clipped signal
    """
    clipped_signal = numpy.copy(signal)
    clipped_signal[numpy.abs(clipped_signal) < threshold] = 0
    if reduce_amplitude:
        clipped_signal -= threshold
    return clipped_signal


if __name__ == "__main__":
    print('Hello world!')
    """
    b) Use the function to center clip a unit sinusoid at -.25 and
    +.25 and estimate the resulting THD graphically using the
    first 6 harmonics (including the fundamental).
    """
    # generate a unit sinusoid
    unit_sinusoid = numpy.sin(2 * numpy.pi * numpy.arange(1000) / 1000)
    # center clip the unit sinusoid at -.25 and +.25
    clipped_unit_sinusoid = center_clip(unit_sinusoid, .25)
    # plot the clipped unit sinusoid
    pyplot.figure()
    pyplot.title('Clipped Unit Sinusoid')
    pyplot.plot(clipped_unit_sinusoid)
    pyplot.xlabel('Samples')
    pyplot.ylabel('Amplitude')
    pyplot.show()

    # calculate the THD of the clipped unit sinusoid
    print(f'THD of clipped unit sinusoid: {total_harmonic_distortion(clipped_unit_sinusoid, 1000)} dBc')
