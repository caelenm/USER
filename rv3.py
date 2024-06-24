import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence
import pandas as pd

label_location = r'C:\Users\Caelen\Documents\GitHub\USER\emotion_vectors'
train_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\train_png'


class CustomGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size, train_directory):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.train_directory = train_directory

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_paths = [os.path.join(self.train_directory, file_name) for file_name in batch_x]

        images = []
        for file_path in batch_x_paths:
            if os.path.exists(file_path):
                image = img_to_array(load_img(file_path, target_size=(150, 150))) / 255.0
                images.append(image)
            else:
                raise FileNotFoundError(f"No such file or directory: '{file_path}'")

        return np.array(images), np.array(batch_y)


def getLabels(label_location):
    labels = []
    for file in os.listdir(label_location):
        vector = []
        file_path = os.path.join(label_location, file)
        with open(file_path, 'r') as f:
            for line in f:
                line = float(line.strip())
                vector.append(line)
        labels.append(vector)
    return labels


def create_dataframe(images_dir, labels):
    dataframe = pd.DataFrame(columns=['image', 'label'])
    for folder in images_dir:
        folder_path = os.path.join(images_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                dataframe = dataframe.append({'image': file, 'label': get_emotion_vector(file)}, ignore_index=True)
    return dataframe


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

    emotion_vector = []
    for f in os.listdir(label_location):
        if emotion_vector_label in filename:
            with open(os.path.join(label_location, f), 'r') as v:
                emotion_vector.extend(v.readlines())
    return emotion_vector


def train(train_data_dir, test_data_dir):
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    datagen = ImageDataGenerator(rescale=1. / 255)
    target_size = (150, 150)
    batch_size = 32

    labels = getLabels(label_location)
    train_filenames = []
    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)
        if os.path.isdir(folder_path):
            train_filenames.extend([os.path.join(folder, file) for file in os.listdir(folder_path)])

    train_generator = CustomGenerator(train_filenames, labels, batch_size, train_directory)

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size
    )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(300))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'cosine_similarity'])

    now = datetime.datetime.now()
    model_path = f'trained/model_{now.strftime("%Y%m%d%H%M%S")}.keras'
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_generator,
        epochs=200,
        validation_data=test_generator,
        verbose=2,
        callbacks=[early_stopping, model_checkpoint]
    )

    print(f'Best model saved to {model_path}')

    return model_path
