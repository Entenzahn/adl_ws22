import matplotlib.pyplot as plt
import librosa

# Print visualization of random song objects
def print_song_info(song):
    fig, axs = plt.subplots(3,2)
    plt.subplots_adjust(hspace=0.8)
    fig.suptitle(f"{song.sid} graphical analysis")
    librosa.display.waveshow(song.samples, ax=axs[0,0])
    librosa.display.specshow(song.qt_spec, ax=axs[1,0])
    librosa.display.specshow(song.qt_spec_base, ax=axs[2,0])
    librosa.display.waveshow(song.samples_30sec, ax=axs[0,1])
    librosa.display.specshow(song.qt_spec_30sec, ax=axs[1,1])
    librosa.display.specshow(song.segment_specs[0].spec, ax=axs[2,1])
    axs[0,0].set_title("Waveform")
    axs[1,0].set_title("Original paper Qt")
    axs[2,0].set_title("Modified Qt")
    axs[0,1].set_title("30sec Waveform")
    axs[1,1].set_title("30sec Qt spectrum")
    axs[2,1].set_title("30sec segmented spectrum")