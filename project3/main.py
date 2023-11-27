import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from scipy.signal import resample, get_window, lfilter


# Helper function to handle windowing and overlap-adding of frames
def frame_processing(signal, frame_size, frame_step, window, process_frame_func, *args, **kwargs):
    num_frames = 1 + int((len(signal) - frame_size) / frame_step)
    processed_signal = np.zeros(len(signal))

    for i in range(num_frames):
        start_index = i * frame_step
        end_index = start_index + frame_size
        frame = signal[start_index:end_index]

        # Window the frame
        windowed_frame = frame * window

        # Process the windowed frame
        processed_frame = process_frame_func(windowed_frame, *args, **kwargs)

        # Overlap-add the processed frame
        processed_signal[start_index:end_index] += processed_frame

    return processed_signal


# process a frame and pitch shift it
def pitch_shift_frame(frame, sr, target_pitch, original_pitch=440.0):
    # Calculate the pitch shift factor
    shift_factor = target_pitch / original_pitch
    # Resample the segment to shift the pitch
    frame_length = len(frame)
    resampled_frame = resample(frame, int(frame_length * shift_factor))
    # Resample back to the original frame length to maintain consistent frame size
    shifted_frame = resample(resampled_frame, frame_length)
    return shifted_frame


# process a frame and generate whispered speech
def whisper_frame(frame, noise_type='white'):
    # LPC order, typically 2 + sr / 1000
    lpc_order = int(2 + len(frame) / 1000)
    # Compute LPC coefficients from the speech frame
    lpc_coeff = librosa.lpc(frame, order=lpc_order)

    # Generate noise
    if noise_type == 'white':
        noise = np.random.randn(len(frame))
    elif noise_type == 'pink':
        noise = np.random.randn(len(frame))
        noise = np.cumsum(noise)
        noise /= max(1, np.sqrt(len(frame)))  # Normalize energy over time
    else:
        raise ValueError("Unsupported noise color: {}. Use 'white' or 'pink'.".format(noise_type))

    # Whisper synthesis using LPC coefficients and generated noise
    whispered_frame = lfilter([1], lpc_coeff, noise)
    return whispered_frame


def part1():
    signal_length = 8000
    ones_signal = np.ones(signal_length)
    sr = 8000
    frame_size_ms, frame_shift_ms = 30, 15
    window_type = 'hann'

    # Convert window parameters from ms to samples
    frame_size = int(frame_size_ms * sr / 1000)
    frame_shift = int(frame_shift_ms * sr / 1000)

    # Get window function
    window = get_window(window_type, frame_size)

    # Process the signal with no additional processing function (pass through)
    synthetic_signal = frame_processing(ones_signal, frame_size, frame_shift, window, lambda x: x)

    # Plot the synthetic signal
    plt.plot(synthetic_signal)
    plt.title('Synthetic Signal with Tapered Windows')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()


def part2():
    sr, signal = wavfile.read('your_voice_sample.wav')
    frame_size_ms, frame_shift_ms = 25, 10
    noise_color = 'pink'  # Can be 'white' or 'pink'

    # Convert frame parameters from ms to samples
    frame_size = int(frame_size_ms * sr / 1000)
    frame_shift = int(frame_shift_ms * sr / 1000)

    # Get the window function
    window = get_window('hann', frame_size)

    # Apply whisper transformation
    whispered_signal = frame_processing(signal, frame_size, frame_shift, window, whisper_frame, noise_type=noise_color)

    # Save the whispered signal to a WAV file
    wavfile.write('whispered_voice.wav', sr, whispered_signal.astype(np.int16))


def part3():
    sr, signal = wavfile.read('your_voice_sample.wav')
    pitch_sequence = [220, 247, 262]  # A simple C-E-G sequence for pitch shifting (in Hz)
    segment_size = 256  # Segment size (in samples)

    # Define the processing for each frame based on the pitch_sequence
    def process_with_pitch_sequence(frame, index):
        current_pitch = pitch_sequence[(index // segment_size) % len(pitch_sequence)]
        return pitch_shift_frame(frame, sr, current_pitch)

    # Process the entire signal frame by frame
    pitched_signal = frame_processing(signal, segment_size, segment_size, np.ones(segment_size),
                                      process_with_pitch_sequence)

    # Save the pitch-modified signal to a WAV file
    wavfile.write('variable_pitch_output.wav', sr, pitched_signal.astype(np.int16))


# Run parts 1, 2, and 3
if __name__ == '__main__':
    part1()
    part2()
    part3()
