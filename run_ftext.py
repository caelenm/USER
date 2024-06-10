import fasttext
import fasttext.util
import numpy as np
import os

#fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('C:/Users/Caelen/Desktop/models/cc.en.300.bin')

# Get the vector of the words: angry, sad, happy, fear, calm, neutral, disgust, surprise
print(ft.get_word_vector('anger'))
print(ft.get_word_vector('sad'))
print(ft.get_word_vector('happy'))
print(ft.get_word_vector('fear'))
print(ft.get_word_vector('calm'))
print(ft.get_word_vector('neutral'))
print(ft.get_word_vector('disgust'))
print(ft.get_word_vector('surprise'))

#save to file
if len(os.listdir('emotion_vectors')) == 0:
        
    np.savetxt('anger.txt', ft.get_word_vector('anger'))
    np.savetxt('sad.txt', ft.get_word_vector('sad'))
    np.savetxt('happy.txt', ft.get_word_vector('happy'))
    np.savetxt('fear.txt', ft.get_word_vector('fear'))
    np.savetxt('calm.txt', ft.get_word_vector('calm'))
    np.savetxt('neutral.txt', ft.get_word_vector('neutral'))
    np.savetxt('disgust.txt', ft.get_word_vector('disgust'))
    np.savetxt('surprise.txt', ft.get_word_vector('surprise'))

