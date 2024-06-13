import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# Set the path to the spectrograms directory
#spectrograms_dir = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms'

# Check if the directory is empty
# if not os.listdir(spectrograms_dir):
#     raise ValueError(f"The directory {spectrograms_dir} is empty. Please check the path.")

# # Split the data into training and testing sets (80% training, 20% testing)
# train_test_split_percentage = 0.8
# train_data, test_data = train_test_split(os.listdir(spectrograms_dir), test_size=1-train_test_split_percentage, random_state=42)

label_location =r'C:\Users\Caelen\Documents\GitHub\USER\emotion_vectors'

class CustomGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return np.array([
            img_to_array(load_img(file_name, target_size=(150, 150))) / 255.0 for file_name in batch_x]), np.array(batch_y)

# for each text file in emotion_vectors, add it to a list of labels
def getLabels(label_location):
    labels = []
    for file in os.listdir(label_location):
        with open(file, 'r') as f:
            for line in f:
                labels.append(line)

def train(train_data_dir, test_data_dir):

    # Define the early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)

    # Create an ImageDataGenerator to preprocess the data
    datagen = ImageDataGenerator(rescale=1./255)

    # Define the target size and batch size for the generator
    target_size = (150, 150)
    batch_size = 32

    labels = getLabels(label_location)


    train_generator = CustomGenerator()

    # # Generate training and testing data using the ImageDataGenerator
    # train_generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=target_size,
    #     class_mode= None,
    #     batch_size=batch_size
    #     #subset='training'
    # )

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        class_mode= None,
        batch_size=batch_size
        #subset='validation'
    )

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(300))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'cosine_similarity'])

    # Define the model checkpoint criteria
    now = datetime.datetime.now()
    model_path = f'trained/model_{now.strftime("%Y%m%d%H%M%S")}.keras'
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=200,
        validation_data=test_generator,
        verbose=2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the trained model to a file
    # import datetime
    # now = datetime.datetime.now()
    # model_path = f'trained/model_{now.strftime("%Y%m%d%H%M%S")}.tf'
    # model.save(model_path, save_format='tf')
    # print(f'Model saved to {model_path}')

    print(f'Best model saved to {model_path}')

    return model_path