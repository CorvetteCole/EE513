import argparse
import numpy
from pathlib import Path
from pyaudio import PyAudio
import logging

# custom log format that includes the logger name, timestamp, and message. Conditional formatting is used to color the
# log level based on severity.
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def save(sound: numpy.ndarray, file: Path, sampling_frequency: int):
    """
    Saves the specified sound to a wav file.

    :param sound: Sound to save as a numpy array
    :param file: Name of the file to save the sound to
    :param sampling_frequency: Sampling frequency in Hz
    """
    # set the file extension to wav if it is not already
    file = file.with_suffix('.wav')

    # convert the sound to a byte array
    sound_bytes = (sound * 127 + 128).astype(numpy.uint8).tobytes()

    # generate the wav header
    header = b'RIFF' + (len(sound_bytes) + 36).to_bytes(4, 'little') + b'WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00' + \
             sampling_frequency.to_bytes(4, 'little') + (sampling_frequency * 1).to_bytes(4, 'little') + \
             b'\x01\x00\x08\x00data' + len(sound_bytes).to_bytes(4, 'little')

    # write the sound to a wav file
    with open(file, 'wb') as f:
        f.write(header)
        f.write(sound_bytes)


class ToneGenerator:

    def __init__(self, reference_frequency: float, scale=6, sampling_frequency=16e3, output_folder=Path('output')):
        """

        :param reference_frequency: Frequency of the base tone in Hz. Default is 440 Hz.
        :param scale: Number of tones in the scale (or octave). Default is 6 for a hexatonic scale.
        :param sampling_frequency: Sampling frequency in Hz. Default is 16 kHz.
        """
        self.scale = scale
        self.reference_frequency = reference_frequency
        self.sampling_frequency = sampling_frequency

        log.info(f'Initializing ToneGenerator with reference frequency {reference_frequency} Hz, scale {scale}, and '
                 f'sampling frequency {sampling_frequency} Hz')

        self.pyaudio = PyAudio()
        self.stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(1), channels=1,
                                        rate=int(sampling_frequency), output=True)
        self.output_folder = output_folder

    def __del__(self):
        self.close()

    def close(self):
        """
        Closes the PyAudio stream.
        """
        log.info('Cleaning up ToneGenerator')
        self.stream.close()
        self.pyaudio.terminate()

    def generate(self, tone_index: int, duration: float) -> numpy.ndarray:
        """
        Generates a tone with the specified frequency, duration, sampling frequency, and index.

        :param tone_index: Index of the tone in the scale. With a hexatonic scale, an index of 0 is the base tone, an
        index of 6 is an octave higher (frequency * 2), and an index of -6 is an octave lower (frequency / 2).
        :param duration: Duration of the tone in seconds
        :param file: Name of the file to save the tone to. If None, the tone will not be saved.
        """
        # calculate the frequency of the tone based on the index and the scale
        frequency = self.reference_frequency * 2 ** (tone_index / self.scale)
        log.debug(
            f'Generating tone with index {tone_index}, frequency {frequency} Hz, and duration {duration} seconds')
        # generate sine wave with the specified frequency and duration at the specified sampling frequency
        return numpy.sin(
            2 * numpy.pi * frequency * numpy.arange(duration * self.sampling_frequency) / self.sampling_frequency)

    def play(self, sound: numpy.ndarray):
        """
        Plays the specified sound using PyAudio.

        :param sound: Sound to play as a numpy array
        """
        log.debug(f'Playing sound with length {len(sound)}')
        self.stream.write((sound * 127 + 128).astype(numpy.uint8).tobytes())

    def save(self, sound: numpy.ndarray, file: str):
        """
        Saves the specified sound to a wav file.

        :param sound: Sound to save as a numpy array
        :param file: Name of the file to save the sound to
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        save(sound, Path(self.output_folder, file), int(self.sampling_frequency))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates musical tones.', epilog='EE513 Project 1')
    parser.add_argument('-f', '--frequency', type=float, required=False, help='Reference frequency of the scale in Hz')
    parser.add_argument('-d', '--duration', type=float, required=False, help='Duration of the tone in seconds')
    parser.add_argument('-sf', '--sampling-frequency', type=float, default=16e3, help='Sampling frequency in Hz')
    parser.add_argument('-s', '--scale', type=int, default=6, help='Number of tones in the scale')
    parser.add_argument('-i', '--tone-index', type=int, default=0, help='Index of the tone')
    parser.add_argument('-o', '--output', type=Path, default=Path('output.wav'), help='Output file name')
    args = parser.parse_args()

    tone_generator = ToneGenerator(args.frequency, args.scale, args.sampling_frequency)

    tone = tone_generator.generate(args.tone_index, args.duration)
    tone_generator.save(tone, args.output)
    tone_generator.play(tone)
