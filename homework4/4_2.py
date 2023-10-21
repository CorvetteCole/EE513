import sounddevice as sd
import soundfile as sf
from pathlib import Path
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

background_noise_file = Path('background_noise.wav')

fs = 8000  # Sample rate
seconds = 10  # Duration of recording

if not background_noise_file.exists():
    noise = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    sf.write('background_noise.wav', noise, fs)
else:
    noise, fs = sf.read(background_noise_file)

sd.play(noise, fs)

mono_recording = np.mean(noise, axis=1)

frequencies, psd = signal.welch(mono_recording, fs)

plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, psd)
plt.title('Average Spectrum Magnitude')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# spectral shape vector
# Frequency bands in Hz
f_hz = [0, 100, 2000, 3000, 4000]
# Corresponding magnitudes
m = [10 ** -7, 10 ** -7, 10 ** -9.5, 10 ** -10, 10 ** -14]
# Normalize frequencies to range from 0 to 1
f = [f / fs for f in f_hz]
numtaps = 400
h = signal.firwin2(numtaps, f, m, fs=fs)

# Apply filter to white noise
white_noise = np.random.normal(0, 1, len(mono_recording))
filtered_noise = signal.lfilter(h, [1.0], noise)
# filtered_noise = np.convolve(white_noise, coeffs)

# Play original white noise
print('Playing original white noise...')
sd.play(white_noise, fs)
sd.wait()

# Play filtered noise
print('Playing filtered white noise...')
sd.play(filtered_noise, fs)
sd.wait()
