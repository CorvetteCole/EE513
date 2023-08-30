import argparse
import numpy
import time
from pathlib import Path
from pyaudio import PyAudio
from typing import Union


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

    def __init__(self, reference_frequency: float, scale=6, sampling_frequency=16e3):
        """

        :param reference_frequency: Frequency of the base tone in Hz. Default is 440 Hz.
        :param scale: Number of tones in the scale (or octave). Default is 6 for a hexatonic scale.
        :param sampling_frequency: Sampling frequency in Hz. Default is 16 kHz.
        """
        self.scale = scale
        self.reference_frequency = reference_frequency
        self.sampling_frequency = sampling_frequency

        self.pyaudio = PyAudio()
        self.stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(1), channels=1,
                                        rate=int(sampling_frequency), output=True)

    def __del__(self):
        self.close()

    def close(self):
        """
        Closes the PyAudio stream.
        """
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
        # generate sine wave with the specified frequency and duration at the specified sampling frequency
        return numpy.sin(
            2 * numpy.pi * frequency * numpy.arange(duration * self.sampling_frequency) / self.sampling_frequency)

    def play(self, sound: numpy.ndarray):
        """
        Plays the specified sound using PyAudio.

        :param sound: Sound to play as a numpy array
        """
        self.stream.write((sound * 127 + 128).astype(numpy.uint8).tobytes())

    def save(self, sound: numpy.ndarray, file: Path):
        """
        Saves the specified sound to a wav file.

        :param sound: Sound to save as a numpy array
        :param file: Name of the file to save the sound to
        """
        save(sound, file, int(self.sampling_frequency))


def test():
    tone_generator = ToneGenerator(261.63, 6, 16e3)
    reference_tone = tone_generator.generate(0, 1)
    fourth_tone = tone_generator.generate(3, 0.25)
    one_octave_below = tone_generator.generate(-6, 0.5)

    # save the tones to wav files
    print('Saving tones to wav files...')
    tone_generator.save(reference_tone, Path('reference_tone.wav'))
    tone_generator.save(fourth_tone, Path('fourth_tone.wav'))
    tone_generator.save(one_octave_below, Path('one_octave_below.wav'))

    print('Graphing tones...')
    import matplotlib.pyplot as plt
    plt.plot(reference_tone)
    plt.plot(fourth_tone)
    plt.plot(one_octave_below)

    # print('Playing reference tone...')
    # tone_generator.play(reference_tone)
    # time.sleep(0.5)
    # print('Playing fourth tone...')
    # tone_generator.play(fourth_tone)
    # time.sleep(0.5)
    # print('Playing one octave below...')
    # tone_generator.play(one_octave_below)
    # time.sleep(1)
    #
    # # generate all tones -6 to 6 and play them
    # for i in range(-6, 7):
    #     tone = tone_generator.generate(i, 1)
    #     print(f'Playing tone {i}...')
    #     tone_generator.play(tone)
    #     time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates musical tones.', epilog='EE513 Project 1')
    parser.add_argument('-f', '--frequency', type=float, required=False, help='Frequency of the tone in Hz')
    parser.add_argument('-d', '--duration', type=float, required=False, help='Duration of the tone in seconds')
    parser.add_argument('-sf', '--sampling-frequency', type=float, default=16e3, help='Sampling frequency in Hz')
    parser.add_argument('-s', '--scale', type=int, default=6, help='Number of tones in the scale')
    parser.add_argument('-i', '--tone-index', type=int, default=0, help='Index of the tone')
    parser.add_argument('-o', '--output', type=Path, default=Path('output.wav'), help='Output file name')
    args = parser.parse_args()

    test()
