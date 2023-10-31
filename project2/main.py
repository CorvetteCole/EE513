import numpy
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path

noise_floor = Path(
    'NoiseFloor20180210.wav')  # ~55 seconds of no signal (ultrasonic transducer not on radial artery): noise floor
tone = Path(
    'RestingHeart20180210.wav')  # ~60 seconds of signal from a person at rest: Doppler signal from a resting heart

fo = 9.1e6  # Frequency of continuous wave ultrasound in Hz
c = 1540.0  # Speed of sound in the body in m/s
theta = 45.0  # Angle of insonifying beam with blood flow direction in degrees


def compute_peak_velocity(max_frequency, fo, c, theta):
    # Calculate the peak velocity using the formula
    v_peak = (max_frequency * c) / (2 * fo * numpy.cos(numpy.deg2rad(theta)))
    return v_peak


def part1():
    """
    Plot the PSD estimate of the background noise signal (noise floor). Present the
    plot with clearly labeled axes. Do the same for the signal plus noise (from person at rest
    data and the recovery from activity). You can put all plots on the same graph (use Matlab’s
    legend command) for a direct comparison.
    """
    # Load the audio data
    fs_noise, noise_data = wavfile.read(noise_floor)
    fs_tone, tone_data = wavfile.read(tone)

    # Compute the PSD estimate using the welch method (a common method)
    f_noise, psd_noise = scipy.signal.welch(noise_data, fs=fs_noise, nperseg=1024)
    f_tone, psd_tone = scipy.signal.welch(tone_data, fs=fs_tone, nperseg=1024)

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_noise, psd_noise, label='Noise Floor', color='blue')
    plt.semilogy(f_tone, psd_tone, label='Resting Heart', color='red')

    # Label the axes and add a legend
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Power Spectral Density (PSD) Estimate')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def part3():
    """
    Compute the spectrogram of blood flow signal from the resting person, where the
    Doppler frequency changes over each individual heart cycle easily observable with
    reasonable resolution. Zoom in so that 2 cycles are clearly seen. From the figure
    determine the highest frequency generated by the blood flow and estimate the peak
    velocity. Must show the zoomed in image with properly labeled axes and what information
    you used from this image to compute the peak and average blood velocity in meters per
    second over the interval selected.

    fo = frequency of continuous wave ultrasound = 9.1 MHz
    c = speed of sound in the body = 1540 m/s
    θ = angle of insonifying beam with blood flow direction ≈ 45°
    """
    # Load the audio data
    fs, tone_data = wavfile.read(tone)

    # Compute the spectrogram
    f, t, Sxx = scipy.signal.spectrogram(tone_data, fs=fs)

    # Zoom in on the spectrogram to select a portion with 2 heart cycles
    t_start = 8
    t_end = 10

    # Find the corresponding frequency bin for the selected interval
    t_idx_start = numpy.where(t >= t_start)[0][0]
    t_idx_end = numpy.where(t <= t_end)[0][-1]

    # Get the spectrogram within the selected interval
    selected_spectrogram = Sxx[:, t_idx_start:t_idx_end]

    # Calculate the peak velocity
    v_peak = compute_peak_velocity(3500, fo, c, theta)

    # Plot the zoomed-in spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(10 * numpy.log10(selected_spectrogram), aspect='auto', cmap='viridis',
               extent=[t[t_idx_start], t[t_idx_end], f[0], f[-1]])
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Zoomed-in Spectrogram')
    # gib more resolution
    plt.ylim(17000, 22000)
    # increase y ticks
    plt.yticks(numpy.arange(17000, 22250, 250))

    # Show the plot
    plt.show()

    print(f"Peak velocity (v_peak): {v_peak} m/s")


def find_average_heart_rate(tone_data, fs):
    # Find peaks in the audio signal
    peaks, _ = scipy.signal.find_peaks(tone_data, height=0.2, distance=int(fs / 2), prominence=0.1, width=1)

    # Calculate the time difference between consecutive peaks
    time_diffs = numpy.diff(peaks) / fs

    # Calculate the average heart rate in Hertz (Hz)
    average_heart_rate = 1 / numpy.mean(time_diffs)

    return average_heart_rate


def part4():
    # Load the audio data
    fs, tone_data = wavfile.read(tone)
    print(f"Average heart rate: {find_average_heart_rate(tone_data, fs)} Hz")


if __name__ == '__main__':
    # part1()
    part3()
    part4()
