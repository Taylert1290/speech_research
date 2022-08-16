import librosa
from playsound import playsound
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

def get_audio(filename):
    x, sr = librosa.load(filename)
    return x, sr

def play_audio(filename):
    playsound(filename)

def visualize_display(filename):
    x, sr = get_audio(filename=filename)
    plt.figure(figsize=(14,5))
    librosa.display.waveshow(x,sr=sr)
    plt.show()

def build_spectrogram(filename):
    x, sr = get_audio(filename=filename)
    #fourrier transfer
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14,5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()

def write_wave(filename):
    x, sample_rate = get_audio(filename=filename)
    # 24 bit soundfile
    sf.write('audio_output/example.wav', x, sample_rate, subtype='PCM_24')

if __name__ == '__main__':
    #x, sr = get_audio(filename='audio_files/Speaker26_001.wav')
    write_wave(filename='audio_files/Speaker26_001.wav')
