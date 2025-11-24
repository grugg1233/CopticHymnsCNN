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