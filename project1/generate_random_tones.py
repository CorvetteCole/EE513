import argparse
from random import randint

from tone_gen import ToneGenerator


def generate_random_tones(number: int, durations: list, indices: list, sampling_frequency: float, scale: int,
                          reference_frequency: float):
    """
    Generates a number of random tones and saves them to a wav file.

    :param number: Number of tones to generate
    :param durations: Durations of the tones to randomly select from
    :param indices: Indices of the notes to randomly select from
    :param sampling_frequency: Sampling frequency in Hz
    :param scale: Number of tones in the scale
    :param reference_frequency: Reference frequency of the scale in Hz
    """
    # generate the tones
    tone_generator = ToneGenerator(reference_frequency, scale, sampling_frequency)

    for i in range(number):
        index = indices[randint(0, len(indices) - 1)]
        duration = durations[randint(0, len(durations) - 1)]
        tone = tone_generator.generate(index, duration)
        tone_generator.play(tone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates random musical tones.', epilog='EE513 Project 1')
    parser.add_argument('-n', '--number', type=int, default=52, help='Number of tones to generate')
    parser.add_argument('-d', '--durations', type=float, nargs='+',
                        default=[2, 1, 3 / 4, 1 / 2, 3 / 8, 1 / 4, 3 / 16, 1 / 8, 1 / 16, 1 / 32],
                        help='Durations of the tones in seconds')
    parser.add_argument('-ni', '--note-indices', type=int, nargs='+', default=[-6, -5, -3, -2, 0, 1, 2, 4, 5, 6],
                        help='Indices of the notes to generate')
    parser.add_argument('-sf', '--sampling-frequency', type=float, default=16e3, help='Sampling frequency in Hz')
    parser.add_argument('-s', '--scale', type=int, default=6, help='Number of tones in the scale')
    parser.add_argument('-f', '--frequency', type=float, default=261.63, help='Reference frequency of the scale in Hz')
    args = parser.parse_args()

    generate_random_tones(args.number, args.durations, args.note_indices, args.sampling_frequency, args.scale,
                          args.frequency)
