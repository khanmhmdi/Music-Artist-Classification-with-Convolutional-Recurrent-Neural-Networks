import torch
import librosa
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from os import walk


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    return plt.show(block=False)


def create_melspectogram(waveform, sample_rate, n_fft=1024, win_length=None,
                         hop_length=512, n_mels=128):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)
    return melspec
    plot_spectrogram(
        melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')


if __name__ == '__main__':
    path = '/content/artist20-baseline/artist20/lists/a20-trn-albums.list'
    data_list = open(path)
    txt = data_list.readlines()
    print(data_list.readlines())
    mel_spectrograms = []

    for album_path in txt:
        # print(album_path)
        album_path = '/content/artist20-mp3s-32k/artist20/mp3s-32k/' + album_path
        # print(album_path)
        album_path = album_path.split('\n')
        f = []
        for (dirpath, dirnames, filenames) in walk(album_path[0] + '/'):
            f.extend(filenames)
            break
        f = [album_path[0] + '/' + i for i in f]
        print(f)
        for songs_path in f:
            waveform, sample_rate = torchaudio.load(songs_path)
            mel_spectrogram = create_melspectogram(waveform=waveform, sample_rate=sample_rate,
                                                   song_name=songs_path.split('/')[-1].split('.')[0])
            mel_spectrograms.append(mel_spectrogram)
