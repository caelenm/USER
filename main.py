def main():
    import os
    #from splitFiles import split_files
    from rv5 import train
    target_directory_sorted = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms_sortedByMood_png'
    train_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\train_png'
    test_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\test_png'

    #converting to spectrograms and organization has already been done

     #split 80/20
    #print('Splitting files...')
    #split_files(target_directory_sorted, train_directory, test_directory)

    #remember to set seed for reproducibility


    train(train_directory, test_directory)
    print("end of main")

if __name__ == '__main__':
    main()