import os

class Config():
    #path information
    ROOT = "output"
    CHECKPOINTS = f"{ROOT}/checkpoints"
    LOGS = f"{ROOT}/logs"
    BEST_MODEL = f"{CHECKPOINTS}/best_hymn_cnn.pt"

    DATA_ROOT = "/kaggle/input/copticsongs/data/hymns"

    #audio features
    SAMPLE_RATE=16_000
    CLIP_SEC = 5.0
    HOP_SEC = 2.5
    N_MELS = 64

    #model 
    N_CLASSES = 4
    DROPOUT_P = 0.5

    #training
    LR = 5e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    SEED = 42 #for reproducibility

    LABEL_MAP = {
        "golgotha": 0,
        "jenainan": 1,
        "taishori": 2,
        "tishori": 3,

    }


    @staticmethod
    def init_dirs():
        os.makedirs(Config.ROOT, exist_ok=True)
        os.makedirs(Config.CHECKPOINTS, exist_ok=True)
        os.makedirs(Config.LOGS, exist_ok=True)
        
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


mp3Golgotha1 = f"{Config.DATA_ROOT}/golgotha/golgotha1.mp3"
mp3Golgotha2 = f"{Config.DATA_ROOT}/golgotha/golgotha2.mp3"
mp3Golgotha3 = f"{Config.DATA_ROOT}/golgotha/golgotha3.mp3"

mp3Jenainan1 = f"{Config.DATA_ROOT}/jenainan/jenainan1.mp3"
mp3Jenainan2 = f"{Config.DATA_ROOT}/jenainan/jenainan2.mp3"
mp3Jenainan3 = f"{Config.DATA_ROOT}/jenainan/jenainan3.mp3"

mp3Taishori1 = f"{Config.DATA_ROOT}/taiShori/taishori1.mp3"
mp3Taishori2 = f"{Config.DATA_ROOT}/taiShori/taishori2.mp3"

mp3Tishori1  = f"{Config.DATA_ROOT}/tishori/tishori1.mp3"
mp3Tishori2  = f"{Config.DATA_ROOT}/tishori/tishori2.mp3"

CLIP_SEC  = Config.CLIP_SEC
HOP_SEC   = Config.HOP_SEC
TARGET_SR = Config.SAMPLE_RATE
N_MELS    = Config.N_MELS


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
    # stores (file_path, waveform, sample_rate)
    loaded_data: List[Tuple[str, torch.Tensor, int]] = []

    print("loading audio files")
    for file_path in all_file_paths:
        p = Path(file_path)
        if not p.exists():
            print(f"WARNING file not found at {file_path}. Skipping.")
            continue
        try:
            waveform, sample_rate = torchaudio.load(str(p))

            # convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != TARGET_SR:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=TARGET_SR
                )
                waveform = resampler(waveform)
                sample_rate = TARGET_SR

            loaded_data.append((file_path, waveform, sample_rate))

            print(
                f"successfully loaded {file_path}, "
                f"(SR: {sample_rate} Hz, Channels: {waveform.shape[0]}, "
                f"Length: {waveform.shape[1]}) frames"
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not loaded_data:
        raise RuntimeError("No audio files loaded successfully.")

    return loaded_data


def build_snippet_index(
    loaded_data: List[Tuple[str, torch.Tensor, int]],
    label_map: Dict[str, int],
) -> List[Tuple[int, int, int, int]]:

    snippetIndex: List[Tuple[int, int, int, int]] = []
    print("\nBuilding audio snippet index")
    for index, (path, waveform, sr) in enumerate(loaded_data):
        num_samples = waveform.shape[1]
        clip_samples = int(CLIP_SEC * sr)
        hop_samples  = int(HOP_SEC * sr)

        if num_samples < clip_samples:
            print(
                f"skipping {path}, too short: "
                f"{num_samples} samples < {clip_samples} clip samples"
            )
            continue

        path_lower = path.lower()
        parts = Path(path_lower).parts
        label_name = parts[-2]  # parent directory

        if label_name not in label_map:
            print(
                f"WARNING: label {label_name} from path {path} "
                f"not found in label_map, skipping file"
            )
            continue
        label_id = label_map[label_name]

        print(f"Processing {path}, (label {label_name}, ID: {label_id})")

        # main sampling loop to iterate over the audio
        start_sample = 0
        while (start_sample + clip_samples) <= num_samples:
            end_sample = start_sample + clip_samples
            snippetIndex.append((index, start_sample, end_sample, label_id))
            start_sample += hop_samples

    print(f"Finished building index. Total snippets: {len(snippetIndex)}")
    return snippetIndex


class HymnSnippetDataset(Dataset):
    def __init__(
        self,
        loaded_data: List[Tuple[str, torch.Tensor, int]],
        snippet_index: List[Tuple[int, int, int, int]],
        sample_rate: int = TARGET_SR,
        n_mels: int = N_MELS,
    ):
        self.loaded_data = loaded_data
        self.snippet_index = snippet_index
        self.sample_rate = sample_rate

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
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

        expected_len = int(CLIP_SEC * self.sample_rate)
        if segment.shape[1] < expected_len:
            pad = expected_len - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, pad))

        mel = self.melspec(segment)
        mel_db = self.to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db, label_id


def loadTheData():
    label_map = Config.LABEL_MAP
    loaded_data = load_all_hymns()
    snippet_index = build_snippet_index(loaded_data, label_map)
    dataset = HymnSnippetDataset(loaded_data, snippet_index)
    return dataset


import torch
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()  


class HymnCNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = 64):
        super().__init__()
        self.n_classes = n_classes

        # Conv stack
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # After conv4 + pooling, we will do global average pooling : 128 features
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Global average pool over freq & time → (B, 128)
        x = x.mean(dim=[2, 3])

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


"""
loaded = load_all_hymns()
label_map = {
    "golgotha": 0,
    "jenainan": 1,
    "taishori": 2,
    "tishori": 3,
}
index = build_snippet_index(loaded, label_map)

ds = HymnSnippetDataset(loaded, index)

loader = DataLoader(ds, batch_size=4, shuffle=True)

mel_batch, labels = next(iter(loader)) 

model = HymnCNN(n_classes = 4)
out = model(mel_batch)

"""


import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio.transforms as T

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split


torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)

Config.init_dirs()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = loadTheData()
len_ds = len(full_dataset)
len_tst = int(0.2 * len_ds)
len_train = len_ds - len_tst

# random split for now
train_ds, val_ds = random_split(
    full_dataset,
    [len_train, len_tst],
    generator=torch.Generator().manual_seed(Config.SEED),
)

train_loader = DataLoader(
    train_ds,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
)

val_loader = DataLoader(
    val_ds,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
)

model = HymnCNN(n_classes=Config.N_CLASSES).to(device)

optimizer = optim.Adam(model.parameters(), lr=Config.LR)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.3
)

global_train_loss = []
global_val_loss = []

count = 0
for epoch in range(Config.NUM_EPOCHS):
    train_running_loss = []
    val_running_loss = []
    start_time = time.time()

    model.train()
    # training loop
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_running_loss.append(loss.item())

    correct = 0
    total = 0

    # validation loop
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            val_running_loss.append(loss.item())

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f"Val Progress --- total: {total}, correct: {correct}")
            print(f"Val Accuracy: {100.0 * float(correct) / float(total):.2f}%")

    global_train_loss.append(sum(train_running_loss) / len(train_running_loss))
    global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

    scheduler.step(global_val_loss[-1])

    print(
        f"epoch [{epoch+1}/{Config.NUM_EPOCHS}], "
        f"TRNLoss: {global_train_loss[-1]:.4f}, "
        f"VALLoss: {global_val_loss[-1]:.4f}, "
        f"Time: {(time.time() - start_time) / 60:.2f} min"
    )

    if (epoch + 1) % 20 == 0:
        count += 1
        MODEL_SAVE_PATH = f"{Config.CHECKPOINTS}/checkpoint{count}.pt"
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'global_trnloss': global_train_loss,
                'global_valloss': global_val_loss,
            },
            MODEL_SAVE_PATH,
        )

plt.plot(range(len(global_train_loss)), global_train_loss, label='TRN loss')
plt.plot(range(len(global_val_loss)), global_val_loss, label='VAL loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training/Validation Loss Plot')
plt.legend()
plt.show()

from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torchaudio
from torch.utils.data import Dataset, Dataloader



mp3Golgotha4 = f"{Config.DATA_ROOT}/golgotha/golgotha4.mp3"
mp3Golgotha5 = f"{Config.DATA_ROOT}/golgotha/golgotha5.mp3"
mp3Jenainan4 = f"{Config.DATA_ROOT}/jenainan/jenainan4.mp3"
mp3Jenainan5 = f"{Config.DATA_ROOT}/jenainan/jenainan5.mp3"
mp3Taishori3 = f"{Config.DATA_ROOT}/taiShori/taishori3.mp3"
mp3Tishori3  = f"{Config.DATA_ROOT}/tishori/tishori3.mp3"


CLIP_SEC = Config.CLIP_SEC
HOP_SEC = Config.HOP_SEC
TARGET_SR = Config.SAMPLE_RATE
N_MELS = Config.N_MELS


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
    label_map = Config.LABEL_MAP
    loaded_data = load_all_hymns_test()
    snippet_index=build_snippet_index_test(loaded_data, label_map)
    dataset = TestHymnSnippetDataset(loaded_data, snippet_index)
    
    return dataset
def evalTest(model, dataloader, device): 
    model.eval() 
    correct = 0
    total = 0 
    num_classes = Config.N_CLASSES
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

    for name, cid in Config.LABEL_MAP.items() :
        if class_total[cid] ==0: 
            print(f"{name:10s}: no snippets in test set")
            continue 
        acc = 100.0*class_correct[cid] / class_total[cid]
        print(f"{name:10s}: {acc:.2f}% ({class_correct[cid]}/{class_total[cid]})")


#main loop 

model = HymnCNN(n_classes=Config.N_CLASSES.to(device))
ckpt = torch.load("/kaggle/working/output/modeloutput/checkpoints/checkpoint5.pt")
model.load_state_dict(ckpt["model_state_dict"])

test_ds = loadTheDataTest() 
test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

evalTest(model, test_loader, device )