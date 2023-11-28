import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from scipy.signal import resample, get_window, lfilter
import argparse
from pathlib import Path


# Helper function to handle windowing and overlap-adding of frames
def frame_processing(signal, frame_size, frame_step, window, process_frame_func, *args, **kwargs):
    # If signal is stereo (2 channels), convert to mono by averaging the channels
    if len(signal.shape) == 2 and signal.shape[1] == 2:
        signal = signal.mean(axis=1)

    num_frames = 1 + int((len(signal) - frame_size) / frame_step)
    processed_signal = np.zeros(len(signal))

    for i in range(num_frames):
        start_index = i * frame_step
        end_index = start_index + frame_size
        frame = signal[start_index:end_index]

        # Window the frame
        windowed_frame = frame * window

        # Process the windowed frame
        processed_frame = process_frame_func(windowed_frame, i, *args, **kwargs)

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
def whisper_frame(frame, frame_index, noise_gain=0.1, gain=16.0):
    # Generate a white noise signal with the same length as the frame
    noise = np.random.randn(len(frame))

    # Generate pink noise
    # noise = np.random.randn(len(frame))
    # noise = np.cumsum(noise)
    # noise /= max(1, np.sqrt(len(frame)))  # Normalize energy over time

    # Pre-emphasize the speech before LPC analysis
    pre_emphasis_coeff = 0.97
    pre_emphasized_frame = np.append(frame[0], frame[1:] - pre_emphasis_coeff * frame[:-1])

    # LPC order, typically 2 + frame_length / 1000
    lpc_order = 32  # int(2 + len(frame) / 1000)
    # Compute LPC coefficients from the speech frame
    lpc_coeff = librosa.lpc(pre_emphasized_frame, order=lpc_order)

    # Apply LPC coefficients to the noise signal
    whispered_frame = lfilter([1], lpc_coeff, noise * noise_gain)

    return whispered_frame * gain


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
    synthetic_signal = frame_processing(ones_signal, frame_size, frame_shift, window, lambda x, y: x)

    # Plot the synthetic signal
    plt.plot(synthetic_signal)
    plt.title('Synthetic Signal with Tapered Windows')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()


def part2():
    sr, signal = wavfile.read(voice_sample_file)
    frame_size_ms, frame_shift_ms = 200, 20

    # Convert frame parameters from ms to samples
    frame_size = int(frame_size_ms * sr / 1000)
    frame_shift = int(frame_shift_ms * sr / 1000)

    # Get the window function
    window = get_window('hann', frame_size)

    # Apply whisper transformation
    whispered_signal = frame_processing(signal, frame_size, frame_shift, window, whisper_frame, gain=768,
                                        noise_gain=0.05)

    # Save the whispered signal to a WAV file
    wavfile.write(voice_sample_file.stem + '_whisper.wav', sr, whispered_signal.astype(np.int16))


def part3():
    sr, signal = wavfile.read(voice_sample_file)
    pitch_sequence = [8, 16, 32, 64, 128, 256]
    frame_size_ms = 1024  # Segment size
    frame_size = int(frame_size_ms * sr / 1000)

    # Get the window function
    window = get_window('hann', frame_size)

    # Define the processing for each frame based on the pitch_sequence
    def process_with_pitch_sequence(frame, frame_index):
        # Determine the pitch for the current frame based on the index
        current_pitch = pitch_sequence[frame_index % len(pitch_sequence)]
        return pitch_shift_frame(frame, sr, current_pitch)

    # Process the entire signal frame by frame
    pitched_signal = frame_processing(signal, frame_size, frame_size, window,
                                      process_with_pitch_sequence)

    # Save the pitch-modified signal to a WAV file
    wavfile.write(voice_sample_file.stem + '_variable_pitch.wav', sr, pitched_signal.astype(np.int16))


# Run parts 1, 2, and 3
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EE513 Project 3')
    parser.add_argument('-f', '--file', type=str, default='hello_world.wav', help='Voice sample file', required=True)
    args = parser.parse_args()

    voice_sample_file = Path(args.file)

    if not voice_sample_file.is_file():
        print(f'File {voice_sample_file} does not exist')
        exit(1)

    print("Part 1")
    part1()
    print(f"Part 2: Turning {voice_sample_file} into a whisper")
    part2()
    print(f"Part 3: Turning {voice_sample_file} into a variable pitch signal")
    part3()
