import os
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import math

import librosa
import librosa.display
import numpy as np 
import pandas as pd 
import IPython.display as ipd
import matplotlib.pyplot as plt

from config import Config as C


mp3Golgotha1 =   f"{C.DATA_ROOT}/golgotha/golgotha1.mp3"
mp3Golgotha2 =   f"{C.DATA_ROOT}/golgotha/golgotha2.mp3"
mp3Golgotha3 =   f"{C.DATA_ROOT}/golgotha/golgotha3.mp3"

mp3Jenainan1 =   f"{C.DATA_ROOT}/jenainan/jenainan1.mp3"
mp3Jenainan2 =   f"{C.DATA_ROOT}/jenainan/jenainan2.mp3"
mp3Jenainan3 =   f"{C.DATA_ROOT}/jenainan/jenainan3.mp3"

mp3Taishori1 =   f"{C.DATA_ROOT}/taiShori/taishori1.mp3"
mp3Taishori2 =   f"{C.DATA_ROOT}/taiShori/taishori2.mp3"

mp3Tishori1  =   f"{C.DATA_ROOT}/tishori/tishori1.mp3"
mp3Tishori2  =   f"{C.DATA_ROOT}/tishori/tishori2.mp3"

CLIP_SEC = C.CLIP_SEC
HOP_SEC = C.HOP_SEC
TARGET_SR = C.SAMPLE_RATE
N_MELS = C.N_MELS

def load_all_hymns() -> List[Tuple[str, torch.Tensor, int]]: 
    all_file_paths = [
        mp3Golgotha1,
        mp3Golgotha2,
        mp3Golgotha3,
        mp3Jenainan1,
        mp3Jenainan2,
        mp3Jenainan3,
        mp3Taishori1,
        mp3Taishori2,
        mp3Tishori1,
        mp3Tishori2,
    ]
    #stores file_path, waveform, sample rate
    loaded_data: List[Tuple[str, torch.Tensor, int]] = []

    print("loading audio files")
    for file_path in all_file_paths: 
        p = Path(file_path)
        if not p.exists():
            print(f"WARNING file not found at {file_path} . skipping. ")
            continue
        try: 
            waveform, sample_rate = torchaudio.load(str(p))

            #convert to mono 
            if waveform.shape[0] > 1: 
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != TARGET_SR:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=TARGET_SR
                )
                waveform = resampler(waveform)
                sample_rate= TARGET_SR

            loaded_data.append((file_path, waveform, sample_rate))

            print(f"successfully loaded {file_path}, (SR: {sample_rate} Hz, Channels: {waveform.shape[0]}, Length: {waveform.shape[1]}) frames")
        except Exception as e: 
            print(f"Error loading {file_path}: {e}")

    if not loaded_data:
        raise RuntimeError("No audio files loaded successfully.")

    return loaded_data

def build_snippet_index(
        loaded_data: List[Tuple[str, torch.Tensor, int]],
        label_map: Dict[str, int]
) -> List[Tuple[int, int, int, int]]:
    
    snippetIndex: List[Tuple[int, int, int, int]] = [] 
    print("\n Building audio snippet index")
    for index, (path, waveform, sr) in enumerate(loaded_data) :
        num_samples=waveform.shape[1]
        clip_samples = int(CLIP_SEC*sr)
        hop_samples = int(HOP_SEC*sr)
        
        if num_samples < clip_samples: 
            print(f"skipping {path}, too short {num_samples} num samples < {clip_samples} clip samples")
            continue

        path_lower = path.lower()
        #.parts comes from pathlib and will split path into parts 
        parts = Path(path_lower).parts
        #folder name
        label_name = parts[-2]

        if label_name not in label_map: 
            print(f"WARNING : label {label_name} from path {path} not found in label_map skipping file")
            continue
        label_id = label_map[label_name]

        print(f"Processing {path}, (label {label_name}, ID: {label_id})")

        # main sampling loop to iteate over the audio 
        start_sample = 0 
        while (start_sample + clip_samples) <= num_samples: 
            end_sample = start_sample+clip_samples
            #keep track of which audio file, from where to where (snippet) , and class
            snippetIndex.append((index, start_sample, end_sample, label_id))
            #iterator
            start_sample += hop_samples

    print(f"Finished building index. Total snippets: {len(snippetIndex)}")
    return snippetIndex

class HymnSnippetDataset(Dataset):
    def __init__(
            self, 
            loaded_data: List[Tuple[str, torch.Tensor, int ]],
            snippet_index: List[Tuple[int, int, int, int]],
            sample_rate : int = TARGET_SR, 
            n_mels : int = N_MELS,
    ): 
        self.loaded_data = loaded_data
        self.snippet_index = snippet_index
        self.sample_rate = sample_rate

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate, 
            n_mels = n_mels,
            n_fft =1024,
            hop_length=256,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.snippet_index)
    
    def __getitem__(self, idx): 
        file_idx, start_sample, end_sample, label_id = self.snippet_index[idx]

        path, waveform, sr = self.loaded_data[file_idx]
        assert sr == self.sample_rate, "sample rate mismatch in dataset."
        segment = waveform[:, start_sample:end_sample]

        expected_len = int (CLIP_SEC*self.sample_rate)
        if segment.shape[1] < expected_len:
            pad = expected_len - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0,pad))

        mel = self.melspec(segment)
        mel_db = self.to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db, label_id

def loadTheData() : 
    label_map = C.LABEL_MAP
    loaded_data = load_all_hymns()
    snippet_index=build_snippet_index(loaded_data, label_map)
    dataset = HymnSnippetDataset(loaded_data, snippet_index)
    return dataset


"""
def main(): 
    label_map = C.LABEL_MAP

    loaded_data = load_all_hymns()

    snippet_index=build_snippet_index(loaded_data, label_map)

    dataset = HymnSnippetDataset(loaded_data, snippet_index)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    mel_batch, labels = batch
    print("\nBatch mel shape:", mel_batch.shape)  # [B, 1, n_mels, T]
    print("Batch labels:", labels)


    #full data visualization 
    sound, sar = torchaudio.load(mp3Tishori1)
    ipd.Audio(data=sound[0,:],rate=sar)
    x, sr =librosa.load(mp3Tishori1)

    plt.figure(figsize=(14, 5))

    librosa.display.waveshow(x, sr=sr)
    X = librosa.stft(x)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14,4))
    plt.show()
    Xdb.shape

if __name__=="__main__":
    main()
    
"""