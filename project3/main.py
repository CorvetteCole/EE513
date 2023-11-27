import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa


def preprocess_speech(signal, input_sr, target_sr=8000, frame_size_ms=30, frame_shift_ms=15, window_type='hann'):
    """
    Process speech signal by windowing and overlap-adding.

    :param signal: Input speech signal array.
    :param input_sr: Input signal sampling rate.
    :param target_sr: Target sampling rate (default 8000 Hz).
    :param frame_size_ms: Window/frame size in milliseconds (default 30 ms).
    :param frame_shift_ms: Window shift in milliseconds (default 10 ms).
    :param window_type: Tapering window type (default 'hann').
    :return: Synthetic signal after processing.
    """

    # Resample if the audio is not already at the desired sampling rate
    if input_sr != target_sr:
        num_samples = int(len(signal) * float(target_sr) / input_sr)
        signal = scipy.signal.resample(signal, num_samples)

    # Convert window size and shift from ms to samples
    frame_size = int(frame_size_ms * target_sr / 1000)
    frame_shift = int(frame_shift_ms * target_sr / 1000)

    # Get the window function
    window = scipy.signal.get_window(window_type, frame_size)

    # Calculate the number of frames from the signal
    num_frames = 1 + int((len(signal) - frame_size) / frame_shift)

    # Initialize an array to store windowed segments
    frames = np.zeros((num_frames, frame_size))

    # Segment the signal into windowed frames
    for i in range(num_frames):
        start_index = i * frame_shift
        end_index = start_index + frame_size
        frames[i, :] = signal[start_index:end_index] * window

    # Overlap-adding frames for synthesis - ensure overlaps add up to 1
    synthetic_signal = np.zeros(len(signal))
    for i in range(num_frames):
        start_index = i * frame_shift
        end_index = start_index + frame_size
        synthetic_signal[start_index:end_index] += frames[i, :]

    return synthetic_signal


def create_whisper(signal, sr, frame_size_ms, frame_shift_ms, noise_color='white'):
    """
    Transform regular speech into a whispering effect.

    :param signal: Input speech signal array.
    :param sr: Sampling rate of the signal.
    :param frame_size_ms: Window/frame size in milliseconds.
    :param frame_shift_ms: Window shift in milliseconds.
    :param noise_color: Color of the noise ('white' or 'pink').
    :return: Whispered speech signal.
    """

    # Convert window sizes from ms to samples
    frame_size = int(frame_size_ms * sr / 1000)
    frame_shift = int(frame_shift_ms * sr / 1000)

    # Initialize whispered signal
    whispered_signal = np.zeros(len(signal))

    # LPC order, typically 2 + sr / 1000
    lpc_order = int(2 + sr / 1000)

    # Process each frame for whisperization
    num_frames = 1 + int((len(signal) - frame_size) / frame_shift)
    for i in range(num_frames):
        start_index = i * frame_shift
        end_index = start_index + frame_size
        if end_index > len(signal):
            break  # Last frame may be shorter than frame_size

        # Pick the current frame
        frame = signal[start_index:end_index]

        # Compute the LPC coefficients from the speech frame
        lpc_coeff = librosa.lpc(frame, order=lpc_order)

        # If desired, find the formant frequencies, although this might not be necessary for whispering
        # formants = find_formant_frequencies(lpc_coef[::-1], sr)

        # Generate noise to replace the prediction error sequence
        if noise_color == 'white':
            noise = np.random.randn(frame_size)
        elif noise_color == 'pink':
            # Generating pink noise is more complex; here we use numpy's random generator for simplicity
            # For a more realistic pink noise, additional filtering may be required
            noise = np.random.randn(frame_size)
            noise = np.cumsum(noise)
            noise /= np.sqrt(frame_size)  # Ensure noise energy is consistent over time
        else:
            raise ValueError("Unsupported noise color: {}. Use 'white' or 'pink'.".format(noise_color))

        # Synthesize the whisper frame using LPC coefficients and generated noise
        whisper_frame = scipy.signal.lfilter([1], lpc_coeff, noise)

        # Overlap-add the whispered frame to the output signal
        whispered_signal[start_index:end_index] += whisper_frame

    return whispered_signal


def pitch_shift_segment(segment, sr, source_pitch, target_pitch):
    """
    Shift the pitch of a signal segment from source_pitch to target_pitch.

    :param segment: Segment of the speech signal to be pitch-shifted.
    :param sr: Sampling rate of the signal.
    :param source_pitch: The original pitch (frequency in Hz) of the segment.
    :param target_pitch: The target pitch (frequency in Hz) to shift to.
    :return: Pitch-shifted segment.
    """
    # Calculate the rate change for the resampling
    if source_pitch == 0 or target_pitch == 0:
        rate_change = 1.0  # Avoid division by zero for unvoiced regions
    else:
        rate_change = source_pitch / target_pitch

    # Resample the segment to the new rate
    resampled_segment = librosa.resample(segment, sr, sr * rate_change)

    # Since the resampling will change the length, we may need to
    # either truncate or zero-pad the segment to make it the right length
    if len(resampled_segment) > len(segment):
        return resampled_segment[:len(segment)]
    else:
        return np.pad(resampled_segment, (0, len(segment) - len(resampled_segment)), 'constant')


def variable_pitch(signal, sr, pitch_sequence, segment_size):
    """
    Modifies the pitch of a signal according to a repeating sequence of pitch values.

    :param signal: Input speech signal.
    :param sr: Sampling rate of the signal.
    :param pitch_sequence: A list of pitch values in Hz, e.g., [220, 247, 262].
    :param segment_size: Size of segments to apply pitch-shifting, typically should match with pitch period.
    :return: Variable pitch signal.
    """
    pitch_sequence = np.array(pitch_sequence)
    pitch_idx = 0  # Index to keep track of which pitch value to apply
    output_signal = []

    # Segment the signal into segment_size chunks
    segments = [signal[x:x + segment_size] for x in range(0, len(signal), segment_size)]

    for segment in segments:
        # Assume the source pitch matches the target pitch in this pitch_sequence
        source_pitch = pitch_sequence[pitch_idx]
        target_pitch = pitch_sequence[pitch_idx]
        pitch_idx = (pitch_idx + 1) % len(pitch_sequence)  # Move to next pitch or loop back

        # Adjust the segment pitch
        pitch_shifted_segment = pitch_shift_segment(segment, sr, source_pitch, target_pitch)

        # Add the pitch-shifted segment to output
        output_signal.extend(pitch_shifted_segment)

    return np.array(output_signal)


# Play original and processed signal (please ensure librosa and simpleaudio packages are installed)
def play_sound(signal, sr):
    sound = sa.play_buffer((signal * 32767).astype(np.int16), 1, 2, sr)
    sound.wait_done()


def part1():
    # Create a signal of all 1's with a length of 8000 (1 second at 8kHz sampling rate)
    signal_length = 8000  # This is adjustable
    ones_signal = np.ones(signal_length)

    # Process the all-ones signal
    # Since the signal is all ones and doesn't need resampling, we use 8000 as both input_sr and target_sr
    # The resulting signal will differ from the input only at the beginning and end due to window tapering
    synthetic_signal = preprocess_speech(ones_signal, input_sr=8000, target_sr=8000, frame_size_ms=30,
                                         frame_shift_ms=15)

    # Plot the synthetic signal
    plt.plot(synthetic_signal, label='Processed Signal with Tapered Windows')
    plt.title('Processed All-Ones Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def part2():
    # Load your voice signal here
    sr, signal = wavfile.read('path_to_your_voice_file.wav')

    # Parameters for the whisper transformation
    frame_size_ms = 25  # window frame size in milliseconds, can be tuned
    frame_shift_ms = 10  # frame shift in milliseconds, can be tuned for overlap
    noise_color = 'pink'  # type of noise, either 'white' or 'pink', can be tuned

    # Apply the whisper transformation
    whispered_signal = create_whisper(signal, sr, frame_size_ms, frame_shift_ms, noise_color)

    # Save the whispered signal to a WAV file for subjective listening test
    wavfile.write('path_to_whispered_voice_file.wav', sr, whispered_signal.astype(np.int16))


def part3():
    sr, signal = wavfile.read('your_voice_sample.wav')
    pitch_sequence = [220, 247, 262]  # A simple C-E-G sequence in Hz
    segment_size = 256  # Size of segments in samples, could be calculated from pitch
    output = variable_pitch(signal, sr, pitch_sequence, segment_size)
    wavfile.write('variable_pitch_output.wav', sr, output.astype(np.int16))


if __name__ == '__main__':
    part1()
