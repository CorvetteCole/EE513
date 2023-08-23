import argparse
import numpy
from pathlib import Path


def generate_tone(frequency: float, duration: float, sampling_frequency: float, tone_index: int) -> numpy.ndarray:
    """
    Generates a tone with the specified frequency, duration, sampling frequency, and index.

    :param frequency: Frequency of the tone in Hz
    :param duration: Duration of the tone in seconds
    :param sampling_frequency: Sampling frequency in Hz
    :param tone_index: Index of the tone in the scale
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates musical tones.', epilog='EE513 Project 1')
    parser.add_argument('-f', '--frequency', type=float, required=True, help='Frequency of the tone in Hz')
    parser.add_argument('-d', '--duration', type=float, required=True, help='Duration of the tone in seconds')
    parser.add_argument('-s', '--sampling-frequency', type=float, default=16e3, help='Sampling frequency in Hz')
    parser.add_argument('-i', '--index', type=int, default=0, help='Index of the tone')
    parser.add_argument('-o', '--output', type=Path, default=Path('output.wav'), help='Output file name')
    args = parser.parse_args()

