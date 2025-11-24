import torch 
import torch.optim as optim 
import torch.nn as nn
import torchaudio.transforms as T

from model import HymnCNN
from config import Config as C
from data import loadTheData

import numpy as np 
import time 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)
if  torch.cuda.is_available(): 
    torch.cuda.manual_seed(C.SEED)

C.init_dirs()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = loadTheData()
len_ds = len(full_dataset)
len_tst = int(0.2*len_ds)
len_train = len_ds - len_tst

#may change to be a manual split instead of using random 
train_ds, val_ds = random_split(
    full_dataset, 
    [len_train, len_tst], 
    generator=torch.Generator().manual_seed(C.SEED), 

)

train_loader = DataLoader(
        train_ds, 
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=C.NUM_WORKERS,
)

val_loader = DataLoader(
        val_ds, 
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,

)

model = HymnCNN(n_classes = C.N_CLASSES).to(device)

optimizer = optim.Adam(model.parameters(), lr=C.LR)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode ='min',patience=5,factor=0.3)

global_train_loss = []
global_val_loss = []

count =0 
for epoch in range(C.NUM_EPOCHS): 
    train_running_loss = [] 
    val_running_loss = [] 
    start_time = time.time() 
    model.train()
    #train
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)): 
        #input and label
        inputs = data[0].to(device)
        labels = data[1].to(device)
        optimizer.zero_grad() 
        #prediction
        output = model(inputs)
        #calculate loss on guess vs actual output 
        loss = criterion(output, labels)
        loss.backward() 
        optimizer.step()

        train_running_loss.append(loss.item())
    correct=0
    total=0

    #val
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            loss=criterion(output, labels)
            val_running_loss.append(loss.item())

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  

            print(f"Val Progress --- total:{total}, correct :{correct}")
            print(f"Val Accuracy : {100.0* float(correct) / float(total):.2f}%")

        global_train_loss.append(sum(train_running_loss) / len(train_running_loss))
        global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

        scheduler.step(global_val_loss[-1])

        print(f"epoch [{epoch+1}/{C.NUM_EPOCHS}], TRNLoss: {global_train_loss[-1]:.4f}, VALLoss: {global_val_loss[-1]:.4f}, Time: {((time.time() - start_time) / 60):.2f}")

        if(epoch + 1) % 20 == 0:
            count+=1
            MODEL_SAVE_PATH = f"{C.CHECKPOINTS}/chckpoint{count}"
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

plt.plot(range(len(global_train_loss)), global_train_loss, label= 'TRN loss')
plt.plot(range(len(global_val_loss)), global_val_loss, label= 'VAL loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training/Validation Loss Plot')
plt.legend()
plt.show()