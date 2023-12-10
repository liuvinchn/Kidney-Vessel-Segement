# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import sys
os.chdir('/kaggle/input/pretrainedmodels-whl')
!pip install --no-index --find-links /kaggle/input/pretrainedmodels-whl pretrainedmodels --no-deps
#!pip install segmentation-models-pytorch
#!pip install pretrainedmodels
#!pip install efficientnet_pytorch

os.chdir('/kaggle/input/seg-whl')
!pip install --no-index --find-links /kaggle/input/seg-whl segmentation-models-pytorch --no-deps

os.chdir('/kaggle/input/eff-whl')
!pip install --no-index --find-links /kaggle/input/eff-whl efficientnet_pytorch --no-deps
#file_list = os.listdir('/opt/conda/lib/python3.10/site-packages/')
#print(file_list)
#!python setup.py install



#import efficientnet_pytorch
import pretrainedmodels
import shutil
import numpy as np
import pandas as pd
from PIL import Image
#from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#import segmentation_models_pytorch as smp
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

train_dataset = VesselDataset('/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense', transform=transform)
train_dataset += VesselDataset('/kaggle/input/blood-vessel-segmentation/train/kidney_1_voi', transform=transform)
train_dataset += VesselDataset('/kaggle/input/blood-vessel-segmentation/train/kidney_3_sparse', transform=transform)
val_dataset = VesselDataset('/kaggle/input/blood-vessel-segmentation/train/kidney_2', transform=transform)

#cache_path = '/kaggle/input/testseg/'
source_path = '/kaggle/input/kidney-vessel-seg/resnet34-333f7ec4.pth'
#if not os.path.exists(cache_path):
#    os.makedirs(cache_path)
#shutil.copy(source_path, cache_path)


checkpoint_path = '/kaggle/input/vesselseg/unet_checkpoint.pth'

#model.load_state_dict(torch.load(checkpoint_path))

criterion = nn.BCEWithLogitsLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

'''
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
    
'''
model_name = "/kaggle/input/model-dir/unet_model.pth"
#torch.save(model, model_name)

model = torch.load(model_name)
model.to(device)



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

submission_df = pd.read_csv('/kaggle/input/blood-vessel-segmentation/sample_submission.csv')
test_data_folders = glob.glob('/kaggle/input/blood-vessel-segmentation/test/*')
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
        print("i ")
        images = images.to(device)
        outputs = model(images)
        predictions = torch.sigmoid(outputs)

        for j in range(predictions.shape[0]):
            rle_mask = rle_encode(predictions[j].cpu().numpy() > 0.5)
            submission_data.append({'id': image_ids[i * 4 + j], 'rle': rle_mask})

submission_df = pd.DataFrame(submission_data)

submission_df['id'] = submission_df['id'].apply(lambda x: x.replace('test/', '').replace('/images/', '_').replace('.tif', ''))
submission_df['id'] = submission_df['id'].apply(lambda x: x.rsplit('_', 1)[0] + '_' + x.rsplit('_', 1)[1].zfill(4))
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("submission generated!")
