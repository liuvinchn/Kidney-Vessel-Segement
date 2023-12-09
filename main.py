# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#!pip module -d path


import os

import pretrainedmodels
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import glob

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VesselDataset(Dataset):
    def __init__(self, data_folder, mode='train', transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.mode = mode
        self.images_folder = os.path.join(data_folder, 'images')
        if self.mode == "train":
            self.labels_folder = os.path.join(data_folder, 'labels')
        self.image_files = os.listdir(self.images_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.image_files[idx])
        if self.mode == "train":
            label_name = os.path.join(self.labels_folder, self.image_files[idx])

        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.mode == "train":
            label = Image.open(label_name).convert('L')

        if self.transform:
            image = self.transform(image)
            if self.mode == "train":
                label = self.transform(label)
        if self.mode == "train":
            return image, label
        else:
            return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = VesselDataset('/Users/lwz/blood-vessel-segmentation/train/kidney_1_dense', transform=transform)
train_dataset += VesselDataset('/Users/lwz/blood-vessel-segmentation/train/kidney_1_voi', transform=transform)
train_dataset += VesselDataset('/Users/lwz/blood-vessel-segmentation/train/kidney_3_sparse', transform=transform)
val_dataset = VesselDataset('/Users/lwz/blood-vessel-segmentation/train/kidney_2', transform=transform)

cache_path = '/Users/lwz/Downloads/VesselSeg/checkpoints/'
source_path = '/Users/lwz/Downloads/VesselSeg/resnet34-333f7ec4.pth'
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
shutil.copy(source_path, cache_path)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
checkpoint_path = '/Users/lwz/Downloads/VesselSeg/unet_checkpoint.pth'

#model.load_state_dict(torch.load(checkpoint_path))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for step, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if step % 1000 == 0:
            print("Step-{},Loss-{}".format(step, loss.item()))

        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation - Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels)

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
torch.save(model.state_dict(), checkpoint_path)

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

submission_df = pd.read_csv('/Users/lwz/blood-vessel-segmentation/sample_submission.csv')
test_data_folders = glob.glob('/Users/lwz/blood-vessel-segmentation/test/*')
image_ids = []


test_dataset = VesselDataset(test_data_folders[0], transform=transform, mode="test")

for folder in test_data_folders[1:]:
    test_dataset += VesselDataset(folder, transform=transform, mode="test")

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

def get_image_index(filename):
    return int(filename.split('.')[0])

for folder in test_data_folders:
    image_files = os.listdir(os.path.join(folder, 'images'))
    image_files.sort()

    folder_name = os.path.basename(folder)

    for i, image_file in enumerate(image_files):
        image_index = get_image_index(image_file)
        image_id = f"{folder_name}_{image_index}"
        image_ids.append(image_id)

submission_data = []

with torch.no_grad():
    for i, (images) in tqdm(enumerate(test_loader), desc="Generating Submission"):
        images = images.to(device)
        outputs = model(images)
        predictions = torch.sigmoid(outputs)

        for j in range(predictions.shape[0]):
            rle_mask = rle_encode(predictions[j].cpu().numpy() > 0.5)
            submission_data.append({'id': image_ids[i * 4 + j], 'rle': rle_mask})

submission_df = pd.DataFrame(submission_data)

submission_df['id'] = submission_df['id'].apply(lambda x: x.replace('test/', '').replace('/images/', '_').replace('.tif', ''))
submission_df['id'] = submission_df['id'].apply(lambda x: x.rsplit('_', 1)[0] + '_' + x.rsplit('_', 1)[1].zfill(4))

submission_df.to_csv('/Users/lwz/submission.csv', index=False)
