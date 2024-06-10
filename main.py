def main():
    import os
    from splitFiles import split_files

    target_directory_sorted = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms_sortedByMood_png'

    #converting to spectrograms and organization has already been done

     #split 80/20
    print('Splitting files...')
    split_files(target_directory_sorted, train_directory, test_directory)