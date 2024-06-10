import os
import shutil
import random

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def split_files(target_directory_sorted, train_directory, test_directory):
   
    # Clear the train and test directories
    clear_directory(train_directory)
    clear_directory(test_directory)
    
    # Loop over each subdirectory in the target directory
    for class_dir in os.listdir(target_directory_sorted):
        class_dir_path = os.path.join(target_directory_sorted, class_dir)

        # Get all the file names for this class
        file_list = os.listdir(class_dir_path)
        num_files = len(file_list)
        num_train = int(0.8 * num_files)

        # Randomly sample files for training, the rest go to testing
        train_files = random.sample(file_list, num_train)
        test_files = list(set(file_list) - set(train_files))

        # Create class-specific directories in train and test directories
        train_class_dir = os.path.join(train_directory, class_dir)
        test_class_dir = os.path.join(test_directory, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move the train and test files into the respective directories
        for file in train_files:
            shutil.copy(os.path.join(class_dir_path, file), os.path.join(train_class_dir, file))
        for file in test_files:
            shutil.copy(os.path.join(class_dir_path, file), os.path.join(test_class_dir, file))