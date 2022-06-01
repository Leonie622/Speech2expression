#Adapt from https://github.com/DimensionNXG/Speech-Emotion-Analyzer
##Inference Script
#importing modules
from keras.models import model_from_json
import librosa
import numpy as np
import pandas as pd
import librosa
json_file = open('speech_emotion/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("speech_emotion/saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")


#define predict_emotion module
def predict_emotion(_audio_path):
    X, sample_rate = librosa.load(_audio_path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    #import pandas as pd
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim, 
                            batch_size=32, 
                            verbose=1)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    #Labels assigned as per README of main repo.
    if liveabc % 5 == 0:
        return "angry"
        # print("Female_angry") # angry
    elif liveabc % 5 == 1: #neutral
        return "neutral" 
    elif liveabc % 5 == 2: #neutral
        return "fear"
    elif liveabc % 5 == 3: #happy
        return "happy"
    else: # elif liveabc % 5 == 4 #sad
        return "sad"
