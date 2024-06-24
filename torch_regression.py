import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

label_location = r'C:\Users\Caelen\Documents\GitHub\USER\emotion_vectors'
all_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\all_png'

# Define the CNN model
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 300)  # 300 is the number of vector dimensions

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class RegressionDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            labels_file (string): Path to the file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels.iloc[idx, 1]  # Assuming the label is in the second column
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_emotion_vector(filename):
    parts = filename.split('-')
    third_number = parts[2]

    emotion_vector_label = None
    if third_number == '05':
        emotion_vector_label = 'angry'
    elif third_number == '02':
        emotion_vector_label = 'calm'
    elif third_number == '07':
        emotion_vector_label = 'disgust'
    elif third_number == '06':
        emotion_vector_label = 'fearful'
    elif third_number == '03':
        emotion_vector_label = 'happy'
    elif third_number == '01':
        emotion_vector_label = 'neutral'
    elif third_number == '04':
        emotion_vector_label = 'sad'
    elif third_number == '08':
        emotion_vector_label = 'surprised'

    return emotion_vector_label

def label_mapping(all_directory, label_location):
    # create a csv with emotion vector labels
    # return the csv file path
    labels = []
    for filename in os.listdir(all_directory):
        if filename.endswith(".png"):
            image_path = os.path.join(all_directory, filename)
            emotion_vector_path = os.path.join(label_location, f"{get_emotion_vector(filename)}.txt")
            with open(emotion_vector_path, "r") as f:
                emotion_vector = f.read().strip()
            labels.append([image_path, emotion_vector])
    
    labels_df = pd.DataFrame(labels, columns=["Image", "Emotion Vector"])
    csv_path = os.path.join(label_location, "labels.csv")
    labels_df.to_csv(csv_path, index=False)
    print(f"Labels saved to {csv_path}")
    return csv_path


def train(all_directory, label_location):

    #split int test and train
    random_seed = 42
    torch.manual_seed(random_seed)
    


    # Create dataset
    dataset = RegressionDataset(
        image_dir=train_directory,
        labels_file=label_mapping(train_directory, label_location),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


#run mapping function
label_mapping(train_directory, label_location)