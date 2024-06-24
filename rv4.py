import os
import logging
from logging.handlers import RotatingFileHandler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence

label_location = r'C:\Users\Caelen\Documents\GitHub\USER\emotion_vectors'
train_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\train_png'
log_file = 'debug_log.txt'

# Set up logging with rotation
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)  # 5 MB per file, keep 2 backups
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[handler])

def getLabels(label_location):
    label_mapping = {}
    for file in os.listdir(label_location):
        vector = []
        file_path = os.path.join(label_location, file)
        with open(file_path, 'r') as f:
            for line in f:
                vector.append(float(line.strip()))
        emotion = file.split('.')[0]  # assuming file is named like 'angry.txt'
        label_mapping[emotion] = vector
    logging.debug(f'Label mapping: {label_mapping}')  # Debug: Log all label mappings
    return label_mapping

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

def create_filename_label_mapping(train_directory, label_mapping):
    filename_label_mapping = {}
    for file in os.listdir(train_directory):
        emotion_vector_label = get_emotion_vector(file)
        if emotion_vector_label in label_mapping:
            filename_label_mapping[file] = label_mapping[emotion_vector_label]
        else:
            logging.error(f"Label not found for file: {file}")
    return filename_label_mapping

class CustomGenerator(Sequence):
    def __init__(self, image_filenames, label_mapping, batch_size, train_directory):
        self.image_filenames = image_filenames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.train_directory = train_directory

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_paths = [os.path.join(self.train_directory, file_name) for file_name in batch_x]
        batch_y = [self.label_mapping.get(file_name, None) for file_name in batch_x]

        # Debugging: Log information about missing labels
        missing_labels = False
        for i, label in enumerate(batch_y):
            if label is None:
                logging.debug(f"Missing label for: {batch_x[i]}")
                batch_y[i] = np.zeros(300)  # Use a default zero vector if not found
                missing_labels = True

        if missing_labels:
            logging.debug(f"Batch {idx} had missing labels that were replaced with zero vectors.")

        images = []
        for file_path in batch_x_paths:
            if os.path.exists(file_path):
                image = img_to_array(load_img(file_path, target_size=(150, 150))) / 255.0
                images.append(image)
            else:
                logging.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"No such file or directory: '{file_path}'")

        images_np = np.array(images)
        labels_np = np.array(batch_y)

        # Check for None values in images and labels
        if None in images_np or None in labels_np:
            logging.error(f"Batch {idx} contains None values")
            logging.error(f"Image paths: {batch_x_paths}")
            logging.error(f"Labels: {batch_y}")
            raise ValueError("Generated batch contains None values")

        if images_np.shape[0] == 0 or labels_np.shape[0] == 0:
            logging.error(f"Batch {idx} has zero length")
            logging.error(f"Image paths: {batch_x_paths}")
            logging.error(f"Labels: {batch_y}")
            raise ValueError("Generated batch has zero length")

        # Additional Debugging
        logging.debug(f'Batch {idx} - Image shapes: {images_np.shape}')
        logging.debug(f'Batch {idx} - Label shapes: {labels_np.shape}')
        logging.debug(f'Batch {idx} - First image path: {batch_x_paths[0]}')
        logging.debug(f'Batch {idx} - First label: {labels_np[0]}')

        return images_np, labels_np

def train(train_data_dir, test_data_dir):
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    datagen = ImageDataGenerator(rescale=1. / 255)
    target_size = (150, 150)
    batch_size = 32

    label_mapping = getLabels(label_location)
    filename_label_mapping = create_filename_label_mapping(train_directory, label_mapping)

    train_filenames = list(filename_label_mapping.keys())
    logging.debug(f'Total training files: {len(train_filenames)}')  # Debug: Log total training files
    logging.debug(f'Sample training file: {train_filenames[0]} with label {filename_label_mapping[train_filenames[0]]}')  # Debug: Log sample file and label

    train_generator = CustomGenerator(train_filenames, filename_label_mapping, batch_size, train_directory)

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size
    )

    model = Sequential()
    model.add(Input(shape=(target_size[0], target_size[1], 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(300))  # Ensure output layer is 300-dimensional for regression

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

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

    logging.debug(f'Best model saved to {model_path}')

    return model_path
