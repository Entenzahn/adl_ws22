import numpy as np
import librosa
import random
import math
from util.AudioData import SegmentSpectrumData


# Processing the samples
def parse_song_data(data):
    for sid in data.keys():
        song = data[sid]
        song.qt_spec = extract_spectograph(song.samples, song.sr)
        song.qt_spec_base = extract_spectograph(song.samples, song.sr, octaves = 3)
        song.samples_30sec = extract_sample(song.samples, song.sr, sec = 30)
        song.qt_spec_30sec = extract_spectograph(song.samples_30sec, song.sr, octaves = 3)
        song.qt_spec_resized = cut_and_fill_spectograph_to_k_col(song.qt_spec, 646)
        song.qt_spec_base_resized = cut_and_fill_spectograph_to_k_col(song.qt_spec_base, 646)
        song.segment_specs = list()
        for i in range(5):
            song.segment_specs.append(
                SegmentSpectrumData(
                    spec = random_segmentation_spectograph(song.qt_spec, target_w=162, segment_w=27),
                    sid = sid
                )
            )
    return data

def extract_spectograph(y, sr, octaves = 8):
    window_length = 8192
    hop_length = window_length // 2
    bins_per_semitone = 2
    bins_per_octave = 12 * bins_per_semitone
    data = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length,
                              fmin=librosa.note_to_hz('C1'),
                              n_bins=bins_per_octave * octaves,
                              bins_per_octave=bins_per_octave))
    return data.astype(np.float16)


def extract_sample(samples:np.ndarray, sr:int, sec:int) -> np.ndarray:
    n_samples_wndw = sr * sec
    n_samples_full = len(samples)
    final_startpoint = n_samples_full - n_samples_wndw
    startpoint = random.sample(range(final_startpoint), k=1)[0]
    endpoint = startpoint + n_samples_wndw
    return samples[startpoint:endpoint]


def cut_and_fill_spectograph_to_k_col(spec:np.ndarray, k:int) -> np.ndarray:
    if spec.shape[1] > k:
        spec = spec[:,:k]
    elif spec.shape[1] < k:
        spec = repeat_spectograph_to_k_col(spec, k)
    return spec


def repeat_spectograph_to_k_col(spec:np.ndarray, k:int) -> np.ndarray:
    orig_l = spec.shape[1]
    diff = k - orig_l
    while True:
        attach_l = min([orig_l, diff])
        spec = np.concatenate((spec, spec[:,0:attach_l]),axis=1)
        diff = diff - attach_l
        if spec.shape[1] == k:
            break
    return spec


def random_segmentation_spectograph(spec:np.ndarray, target_w:int, segment_w:int) -> np.ndarray:
    num_passes = math.ceil(target_w/segment_w)
    segmented_spec = np.empty(shape=(spec.shape[0],target_w))
    final_startpoint = spec.shape[1] - segment_w
    for i in range(num_passes):
        startpoint = random.sample(range(final_startpoint),k=1)[0]
        endpoint = startpoint + segment_w
        new_segment = spec[:,startpoint:endpoint]
        l_col = i*segment_w
        r_col = (i+1)*segment_w
        segmented_spec[:,l_col:r_col] = new_segment
    return segmented_spec
