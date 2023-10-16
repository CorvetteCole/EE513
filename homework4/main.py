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
    f, pxx = scipy.signal.periodogram(signal, fs=sampling_frequency, window=('kaiser', 38), scaling='spectrum',
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
        # can't just subtract the threshold because the signal is centered at 0
        clipped_signal[clipped_signal > 0] -= threshold
        clipped_signal[clipped_signal < 0] += threshold
    return clipped_signal


def generate_unit_sinusoid(frequency: float, sampling_frequency: float, duration: float):
    """
    Generates a unit sinusoid with the specified frequency, sampling frequency, and duration.

    :param frequency: Frequency of the unit sinusoid in Hz
    :param sampling_frequency: Sampling frequency of the unit sinusoid in Hz
    :param duration: Duration of the unit sinusoid in seconds
    :return: The unit sinusoid
    """
    return numpy.sin(2 * numpy.pi * frequency * numpy.arange(duration * sampling_frequency) / sampling_frequency)


if __name__ == "__main__":
    print('Hello world!')
    """
    b) Use the function to center clip a unit sinusoid at -.25 and
    +.25 and estimate the resulting THD graphically using the
    first 6 harmonics (including the fundamental).
    """

    sampling_frequency = 16e3

    # generate a unit sinusoid
    unit_sinusoid = generate_unit_sinusoid(100, sampling_frequency, 0.05)

    # center clip the unit sinusoid at -.25 and +.25
    clipped_unit_sinusoid = center_clip(unit_sinusoid, .25, reduce_amplitude=True)

    # plot the original and clipped unit sinusoid on the same plot
    time_axis = numpy.arange(len(unit_sinusoid)) / sampling_frequency
    pyplot.figure()
    pyplot.title('Unit sinusoid and center clipped unit sinusoid')
    pyplot.plot(time_axis, unit_sinusoid, label='Unit sinusoid')
    pyplot.plot(time_axis, clipped_unit_sinusoid, label='Center clipped unit sinusoid')
    pyplot.xlabel('Time [seconds]')
    pyplot.xlim(0, 0.05)
    pyplot.ylabel('Amplitude')
    pyplot.legend()
    pyplot.show()

    # calculate the THD of the clipped unit sinusoid
    print(f'THD of clipped unit sinusoid: {total_harmonic_distortion(clipped_unit_sinusoid, 1000)} dBc')
