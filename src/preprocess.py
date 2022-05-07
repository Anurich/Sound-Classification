import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch import sign
import config
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from torch.utils.data import Dataset

def readCSV():
    def combineFile(x1, x2):
        return os.path.join(config.AUDIO_PATH,x1, x2)
    def wordindex(x1, x2):
        word2index = dict()
        for x, y in zip(x1, x2):
            if word2index.get(x) == None:
             word2index[x] = y
        index2word = {value:key for key, value in word2index.items()}
        return index2word, word2index
    
    def dumpPickle(dictionary, filepath):
        with open(filepath, "wb") as fp:
            pickle.dump(dictionary, fp)

    df = pd.read_csv(config.CSV_FILE_PATH)
    df["path"] = "fold"
    df["path"] = df["path"].astype(str) + df["fold"].astype(str)
    df["path"] = df.apply(lambda x: combineFile(x["path"], x["slice_file_name"]), axis=1)    
    train, test = train_test_split(df, test_size=0.3, stratify=df["classID"])
    train, dev  = train_test_split(train, test_size=0.2, stratify=train["classID"])

    class_id = list(train["classID"].values) 
    class_name = list(train["class"].values)
    index2word, word2index  = wordindex(class_name, class_id)

    # let's save train dev and test file 
    train.to_csv(config.CSV_TRAIN_FILE_PATH, index=False)
    test.to_csv(config.CSV_TEST_FILE_PATH, index=False)
    dev.to_csv(config.CSV_DEV_FILE_PATH,  index=False)
    # saving the pickle
    dumpPickle(word2index, config.PICKLE_FILE_PATH_w2i)
    dumpPickle(index2word, config.PICKLE_FILE_PATH_i2w)

class extractFeatures:
    def __init__(self, filepath, picklepath, savepath) -> None:
        self.word2index = pickle.load(open(picklepath, "rb"))
        self.features = []
        self.labels   = []
        self.df = pd.read_csv(filepath)[["path", "class"]]
        self.savepath =savepath
        print(self.savepath)
        

    def savePickle(self, pathname, x):
        with open(pathname, "wb") as fp:
            pickle.dump(x, fp)

        
    def truncation(self, x):
        if x.shape[0] > config.NUM_SAMPLES:
            x = x[:config.NUM_SAMPLES]
        return x 
    
    def padding(self, x):
        if x.shape[0] < config.NUM_SAMPLES:
            left_to_pad = config.NUM_SAMPLES - x.shape[0]
            padd = [0]*left_to_pad
            x = list(x) + padd
        return np.array(x)

    @property
    def featExtraction(self):
        data = {
            "X": [],
            "Y": []
        }
        
        for idx, row in tqdm(self.df.iterrows()):
            signal, sr = librosa.load(row["path"], sr=config.SAMPLE_RATE)
            # let's compute the MFCC 
            signal = self.truncation(signal)
            signal = self.padding(signal)
            signal = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc= config.N_MFCC , n_fft = config.N_FFTT, hop_length =config.HOP_LENGTH)
            signal = signal.T
            data["X"].append(signal.tolist())
            data["Y"].append(self.word2index[row["class"]])
        
        self.savePickle(self.savepath, data)
        
exc = extractFeatures(config.CSV_TRAIN_FILE_PATH, config.PICKLE_FILE_PATH_w2i,config.FEAT_PATH_TRAIN)
exc.featExtraction
exc = extractFeatures(config.CSV_DEV_FILE_PATH, config.PICKLE_FILE_PATH_w2i,config.FEAT_PATH_DEV)
exc.featExtraction
exc = extractFeatures(config.CSV_TEST_FILE_PATH, config.PICKLE_FILE_PATH_w2i,config.FEAT_PATH_TEST)
exc.featExtraction
