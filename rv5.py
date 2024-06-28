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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

label_location = r'C:\Users\Caelen\Documents\GitHub\USER\emotion_vectors'
all_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms_sortedByMood_png'
train_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\train_png_original'
test_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\test_png_original'
log_file = 'debug_log.txt'

# Set up logging with rotation
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)  # 5 MB per file, keep 2 backups
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[handler])

def getLabels(label_location):
    label_mapping = {}
    for file in os.listdir(label_location):
        file_path = os.path.join(label_location, file)
        vector = np.loadtxt(file_path)
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

def create_filename_label_mapping(all_directory, label_mapping):
## label is one of options in get_emotion_vector
## missing labels are not an option
    filename_label_mapping = {}
    for folder in os.listdir(all_directory):
        filenames = [f for f in os.listdir(os.path.join(all_directory, folder))]
        for f in filenames:
            emotion_vector_label = get_emotion_vector(f)
            filename_label_mapping[f] = label_mapping[emotion_vector_label] #os.basename(f) in place of [f]?
            logging.debug(f'Filename label mapping: {filename_label_mapping}')  # Debug: Log all filename label mappings
    return filename_label_mapping

class CustomTrainGenerator(Sequence):
    def __init__(self, image_filenames, label_mapping, batch_size, train_directory):
        self.image_filenames = image_filenames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.train_directory = train_directory

    def __len__(self):
        ## returns the number of batches per epoch
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
       ## yields batches of images and labels
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = [self.label_mapping[filename] for filename in batch_x]
        batch_x_paths = [os.path.join(self.train_directory, get_emotion_vector(file_name) + "_png", file_name) for file_name in batch_x]

        images = []
        for file_path in batch_x_paths:
            image = img_to_array(load_img(file_path, target_size=(150, 150))) / 255.0
            images.append(image)
        
        return np.array(images), np.array(batch_y)

class CustomTestGenerator(Sequence):
    def __init__(self, image_filenames, label_mapping, batch_size, test_directory):
        self.image_filenames = image_filenames
        self.label_mapping = label_mapping
        self.batch_size = batch_size
        self.test_directory = test_directory

    def __len__(self):
        ## returns the number of batches per epoch
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
       ## yields batches of images and labels
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = [self.label_mapping[filename] for filename in batch_x]
        batch_x_paths = [os.path.join(self.test_directory, get_emotion_vector(file_name) + "_png", file_name) for file_name in batch_x]

        images = []
        for file_path in batch_x_paths:
            image = img_to_array(load_img(file_path, target_size=(150, 150))) / 255.0
            images.append(image)
        
        return np.array(images), np.array(batch_y)

def train():
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    datagen = ImageDataGenerator(rescale=1. / 255)
    target_size = (150, 150)
    batch_size = 32
    print("a")
    label_mapping = getLabels(label_location)
    #filename_label_mapping = create_filename_label_mapping(all_directory, label_mapping)
    # with open('~/filename_label_mapping.pkl', 'wb') as f:
    #     pickle.dump(filename_label_mapping, f)
    # if pickle does not exist, create it
    if not os.path.exists('filename_label_mapping.pkl'):
        filename_label_mapping = create_filename_label_mapping(all_directory, label_mapping)
        with open('filename_label_mapping.pkl', 'wb') as f:
            #save
            pickle.dump(filename_label_mapping, f)
    with open('filename_label_mapping.pkl', 'rb') as f:
        #load
        filename_label_mapping = pickle.load(f)
    print("b")
    all_filenames = list(filename_label_mapping.keys())
    #make random seed
    np.random.seed(42)
    # take 80 percent of each subfolder for training, 20 percent for testing
                # for folder in os.listdir(all_directory):
                #     filenames = [f for f in os.listdir(os.path.join(all_directory, folder)) if f.endswith('.png')]
                #     np.random.shuffle(filenames)
                #     split_index = int(0.8 * len(filenames))
                #     train_filenames = filenames[:split_index]
                #     test_filenames = filenames[split_index:]
    train_filenames=[f for f in os.listdir(train_directory) if f.endswith('.png')]
    test_filenames=[f for f in os.listdir(test_directory) if f.endswith('.png')]
    print(f'Total training files: {len(train_filenames)}')
    print(f'total test files: {len(test_filenames)}')
  #  logging.debug(f'Total training files: {len(train_filenames)}')  # Debug: Log total training files
    #logging.debug(f'Sample training file: {train_filenames[0]} with label {filename_label_mapping[train_filenames[0]]}')  # Debug: Log sample file and label

   
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

    #filename label mapping for training and testing
    #init separate mappings for each
    train_filename_label_mappings = {}
    test_filename_label_mappings = {}
    # add files from train_png_original and test_png_original to the mappings
    for key, value in filename_label_mapping.items():
        if key in train_filenames:
            train_filename_label_mappings[key] = value
        elif key in test_filenames:
            test_filename_label_mappings[key] = value
            # for key, value in filename_label_mapping.items():
            #     if key in train_filenames:
            #         train_filename_label_mappings[key] = value
            #     elif key in test_filenames:
            #         test_filename_label_mappings[key] = value
    

    history = model.fit(
        x = CustomTrainGenerator(train_filenames, train_filename_label_mappings, batch_size, train_directory),
        epochs=200,
        # use_multiprocessing=True,
        # workers=8,
        validation_data=CustomTestGenerator(test_filenames, test_filename_label_mappings, batch_size, test_directory),
        verbose=2,
        callbacks=[early_stopping, model_checkpoint]
    )

    logging.debug(f'Best model saved to {model_path}')
    print(f'Best model saved to {model_path}')

    return model_path
