import os
import librosa
import numpy as np
import json
import sys


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


def cut_and_fill_spectograph_to_k_col(spec:np.ndarray, k:int) -> np.ndarray:
    if spec.shape[1] > k:
        spec = spec[:,:k]
    elif spec.shape[1] < k:
        spec = repeat_spectograph_to_k_col(spec, k)
    return spec


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


def export_tensor(audio_file, json_file):
    data = dict()

    # checking if it is a file
    if os.path.isfile(audio_file):
        try:
            y, sr = librosa.load(audio_file)
        except:
            print(f"ERROR: Could not open {audio_file}")
        print(f" Extracting {audio_file}...")

    full_spec = extract_spectograph(y,sr)
    if full_spec.shape[1] != 646:
        spec = cut_and_fill_spectograph_to_k_col(full_spec, 646).tolist()
    else:
        spec=full_spec.tolist()

    with open(json_file, "w") as f:
        json.dump(spec, f)
    print(f"Tensor exported to {json_file}")


if __name__ == "__main__":
    audio_file = sys.argv[1]
    if len(sys.argv) > 2:
        json_file = sys.argv[2]
    else:
        json_file = "./tensor_export.json"
    export_tensor(audio_file, json_file)