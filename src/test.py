import config
import pickle
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd
from customdataloader import dataset
from torch.utils.data import DataLoader
from model import MLP, CNNetwork
if __name__ == "__main__":
    def readPickle(filename):
        return pickle.load(open(filename, "rb"))
    
    index2word = readPickle(config.PICKLE_FILE_PATH_i2w)
    numclasses = len(index2word.keys())
    md = CNNetwork(numclasses)

    test = config.CSV_TEST_FILE_PATH
    test_feat = config.FEAT_PATH_TEST
    test_dataset  =  dataset(config.FEAT_PATH_TEST)
    testloader  = DataLoader(test_dataset, batch_size= 8, shuffle=True)

    df_test = pd.read_csv(test)[["path", "class"]]
    
    true_y = []
    pred_y = []
    md.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            x, y = data
            x = x.unsqueeze(1)
            output = md(x)
            output = torch.log_softmax(output,-1)
            _, index = torch.max(output, -1)

            pred_y.extend(index.detach().cpu().tolist())
            true_y.extend(y.detach().cpu().tolist())
            
        print(accuracy_score(true_y, pred_y))


    
