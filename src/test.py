from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torchaudio
from torch.utils.data import Dataset, Dataloader

from config import Config as C 
from model import HymnCNN

mp3Golgotha4 = f"{C.DATA_ROOT}/golgotha/golgotha4.mp3"
mp3Golgotha5 = f"{C.DATA_ROOT}/golgotha/golgotha5.mp3"
mp3Jenainan4 = f"{C.DATA_ROOT}/jenainan/jenainan4.mp3"
mp3Jenainan5 = f"{C.DATA_ROOT}/jenainan/jenainan5.mp3"
mp3Taishori3 = f"{C.DATA_ROOT}/taiShori/taishori3.mp3"
mp3Tishori3  = f"{C.DATA_ROOT}/tishori/tishori3.mp3"


CLIP_SEC = C.CLIP_SEC
HOP_SEC = C.HOP_SEC
TARGET_SR = C.SAMPLE_RATE
N_MELS = C.N_MELS


def load_all_hymns_test() -> List[Tuple[str, torch.Tensor, int]]: 

    TEST_FILES = [
        mp3Golgotha4,
        mp3Golgotha5,
        mp3Jenainan4,
        mp3Jenainan5,
        mp3Taishori3,
        mp3Tishori3,
    ]
    #stores file_path, waveform, sample rate
    loaded_data: List[Tuple[str, torch.Tensor, int]] = []

    print("loading audio files")
    for file_path in TEST_FILES: 
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


def build_snippet_index_test(
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

class TestHymnSnippetDataset(Dataset):
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

def loadTheDataTest() : 
    label_map = C.LABEL_MAP
    loaded_data = load_all_hymns_test()
    snippet_index=build_snippet_index_test(loaded_data, label_map)
    dataset = TestHymnSnippetDataset(loaded_data, snippet_index)
    
    return dataset
def evalTest(model, dataloader, device): 
    model.eval() 
    correct = 0
    total = 0 
    num_classes = C.N_CLASSES
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for c in range(num_classes):
                mask = (labels==c)
                class_total[c] += mask.sum().item() 
                class_correct[c] += ((preds == labels) & mask).sum().item()

    overall_acc = 100.0 * correct / total if total > 0 else 0.0

    print(f"\nRESULTS\n")
    print(f"Total snippets: {total}")
    print(f"Overall accuracy: {overall_acc:.2f}%\n")

    for name, cid in C.LABEL_MAP.items() :
        if class_total[cid] ==0: 
            print(f"{name:10s}: no snippets in test set")
            continue 
        acc = 100.0*class_correct[cid] / class_total[cid]
        print(f"{name:10s}: {acc:.2f}% ({class_correct[cid]}/{class_total[cid]})")
