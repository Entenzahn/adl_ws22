import os
import re
import pandas as pd
import pickle
from pydub import AudioSegment
import librosa
from util.AudioData import SongData


def convert_mp3_to_wav(folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            song_id = re.sub('\.LOFI\.mp3','', filename)
            name_stem = re.sub('\.mp3','', filename)
            print(f"Converting {song_id} at {f}", end="\r")
            sound = AudioSegment.from_mp3(f)
            sound.export(target_folder+name_stem+".wav", format="wav")


def load_audio_data(folder, key_df, filetype="wav"):
    data = dict()
    not_loadable = []

    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            song_id = re.sub(f'\.LOFI\.{filetype}','', filename)
            if song_id in key_df.index:
                print(f"Loading song {song_id}", end="\r")
                try:
                    y, sr = librosa.load(f)
                    data[song_id] = SongData()
                    data[song_id].samples = y
                    data[song_id].sr = sr
                    data[song_id].sid = song_id
                except:
                    not_loadable.append(song_id)
                    continue
    print(f"Managed to load {100 - (len(not_loadable) / (len(not_loadable) + len(data.keys()))*100)}%")
    return data


def load_giantsteps_keys(folder):
    keys = {'song_id': [], 'key': []}

    # Loading the annotations
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f) as f:
                song_id = re.sub('\.LOFI\.key', '', filename)
                key = f.read()
                keys['song_id'].append(song_id)
                keys['key'].append(key)

    # Parsing the annotations into a dataframe
    giantsteps_keys = pd.DataFrame.from_dict(keys).set_index('song_id')
    return giantsteps_keys


def load_giantsteps_mtg_keys(file):
    key_equivalents = {re.compile('^G#'): 'Ab',
                       re.compile('^A#'): 'Bb',
                       re.compile('^C#'): 'Db',
                       re.compile('^D#'): 'Eb',
                       re.compile('^F#'): 'Gb'}

    giantsteps_mtg_keys = pd.read_csv(file, sep='\t').set_index('ID').drop('C', axis=1)
    giantsteps_mtg_keys = giantsteps_mtg_keys.rename({'MANUAL KEY': 'key'}, axis=1).replace(key_equivalents, regex=True)
    giantsteps_mtg_keys.index = giantsteps_mtg_keys.index.map(str)
    return giantsteps_mtg_keys


def store_pickle(data, pickle_filepath):
    fileObj = open(pickle_filepath, 'wb')
    pickle.dump(data,fileObj)
    fileObj.close()


def load_pickle(pickle_filepath):
    fileObj = open(pickle_filepath, 'rb')
    data = pickle.load(fileObj)
    fileObj.close()
    return data

