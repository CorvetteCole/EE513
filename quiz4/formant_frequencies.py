import numpy
from typing import List


def find_formant_frequencies(coefficients: List[float], sampling_rate=16e3) -> List[float]:
    """
    Finds the formant frequencies of the input coefficients of the LPC polynomial in reverse order.

    :param coefficients:  The coefficients of the LPC polynomial in reverse order
    :param sampling_rate:  The sampling rate of the signal in Hz
    :return: The formant frequencies in Hz
    """
    roots = numpy.roots(coefficients)

    formant_frequencies = []
    # only consider roots with positive imaginary parts
    for root in roots:
        if numpy.imag(root) > 0:
            angle = numpy.angle(root)
            frequency = (angle / (2 * numpy.pi)) * sampling_rate
            formant_frequencies.append(frequency)

    formant_frequencies.sort()
    return formant_frequencies


if __name__ == "__main__":
    print(find_formant_frequencies([0.3965, 0.772, 1.46, 1.2, 1]))
