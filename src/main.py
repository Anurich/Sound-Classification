from glob import glob
import config 
from model import MLP, CNNetwork
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
from tqdm import tqdm 
from torch.utils.data import DataLoader
from customdataloader import dataset
import pickle
all_previous_loss = []
counter = 0
if __name__ == "__main__":
    def readPickle(filename):
        return pickle.load(open(filename, "rb"))

    def earlystopping(devloss):
        global counter 
        global all_previous_loss
        if counter != 5:
            # we will store the value 
            all_previous_loss.append(devloss.item())
            counter +=1
        
        else:
            index = np.argmax(all_previous_loss)
            if index != 0:
                # it means that loss is started increasing 
                return True
            else:
                all_previous_loss.pop(0)
                counter -=1




    # read index2word file
    index2word = readPickle(config.PICKLE_FILE_PATH_i2w)
    numClass   = len(index2word.keys())
    #construction of dataset
    train_dataset =  dataset(config.FEAT_PATH_TRAIN)
    dev_dataset   =  dataset(config.FEAT_PATH_DEV)
   
    # construction fo dataloader 
    trainloader = DataLoader(train_dataset, batch_size= config.BATCH_SIZE, shuffle=True)
    devloader   = DataLoader(dev_dataset, batch_size= config.BATCH_SIZE, shuffle=True)
    
    #md = MODEL(numClass)
    md = CNNetwork(numClass)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(md.parameters(), lr =config.LR, weight_decay=0.001)

    for i in tqdm(range(config.ITERATION)):
        trian_pred = []
        train_true = []
        total_train_loss = []
        for traindata in trainloader:
            x, y = traindata
            train_true.extend(y.detach().cpu().tolist())
            x = x.unsqueeze(1)
            optimizer.zero_grad()
            output = md(x)
            prediction  = torch.log_softmax(output,-1)
            _, predindex = torch.max(prediction,-1)
            trian_pred.extend(predindex.detach().cpu().tolist())
            lossfn = loss(output, y)
            lossfn.backward()
            optimizer.step()

            total_train_loss.append(lossfn.item())

        if i %10 == 0 and i!=0:
            md.eval()
            total_dev_loss = []
            dev_true =[]
            dev_pred =[]
            with torch.no_grad():
                for devdata in devloader:
                    x, y = devdata
                    dev_true.extend(y.detach().cpu().tolist())
                    x = x.unsqueeze(1)
                    output = md(x)
                    prediction  = torch.log_softmax(output,-1)
                    _, predindex = torch.max(prediction,-1)
                    dev_pred.extend(predindex.detach().cpu().tolist())

                    lossdev = loss(output, y)
                    total_dev_loss.append(lossdev.item())
            
            # we pass the mean value 
            devloss_mean = np.mean(total_dev_loss)
            boolval = earlystopping(devloss_mean)
            if boolval:
                break

            print(f"Train loss {np.mean(total_train_loss)} and Train accuracy {accuracy_score(train_true, trian_pred)}")
            print(f"Dev loss {np.mean(total_dev_loss)} and Dev accuracy {accuracy_score(dev_true, dev_pred)}")
    
    torch.save({
        "model_state_dict": md.state_dict()
    }, config.MODEL_SAVED)



