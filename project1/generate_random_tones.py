import argparse

from project1.tone_gen import generate_tone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates random musical tones.', epilog='EE513 Project 1')
    parser.add_argument('-n', '--number', type=int, default=52, help='Number of tones to generate')
    args = parser.parse_args()
