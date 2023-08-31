import argparse
import numpy

import matplotlib.pyplot as pyplot
from tone_gen import ToneGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates a musical scale over two octaves.',
                                     epilog='EE513 Project 1')
    parser.add_argument('-d', '--duration', type=float, default=1 / 4,
                        help='Durations of the tones in seconds')
    parser.add_argument('-sf', '--sampling-frequency', type=float, default=16e3, help='Sampling frequency in Hz')
    parser.add_argument('-s', '--scale', type=int, default=6, help='Number of tones in the scale')
    parser.add_argument('-f', '--frequency', type=float, default=261.63, help='Reference frequency of the scale in Hz')
    args = parser.parse_args()

    tone_generator = ToneGenerator(args.frequency, args.scale, args.sampling_frequency)

    # set the plot parameters so the sine wave is visible (instead of filling the entire plot)
    pyplot.ylim(-1.1, 1.1)
    pyplot.xlim(0, 100)
    pyplot.legend([f'{i}' for i in range(-6, 7)])

    # generate all tones -6 to 6
    tones = []
    for i in range(-6, 7):
        tone = tone_generator.generate(i, args.duration)
        pyplot.plot(tone)
        tones.append(tone)

    scale = numpy.concatenate(tones)
    tone_generator.save(scale,
                        f'scale_reference_frequency_{args.frequency}_sampling_frequency_{args.sampling_frequency}_scale_{args.scale}')
    tone_generator.play(scale)
    pyplot.show()
