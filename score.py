#!/usr/bin/env python

import pandas as pd
import numpy as np
import math
import sys



def scoring(truthcsv,predcsv):
    truth = pd.read_csv(truthcsv, header = None, index_col = 0)
    predict = pd.read_csv(predcsv, header = None, index_col = 0)
    
    tind = truth.index.values
    pind = predict.index.values
    
    truth.index = [i.split(".")[0] for i in tind.tolist()]
    
    score = 0.0
    
    numimg = len(tind.tolist())
    numcat = len(truth.columns.values.tolist())
    
    if(set(tind.tolist())!= set(pind.tolist())):
        if(len(tind) != len(pind)):
            sys.exit("ERROR: Incorrect number of images in prediction file; found " + str(len(pind)) + " but expected " + str(len(tind)))
        else: 
            sys.exit("ERROR: Unknown images in prediction file: " + str(np.setdiff1d(pind, tind)))
            
    
    for img in tind:
        val = predict.loc[img].values
        
	if np.isnan(val).any():
            sys.exit("ERROR: Incorrect number of probabilities for image " + str(img) + "\n")
        
        if (max(val) > 1 or min(val) < 0):
            sys.exit("ERROR: Invalid probabilities for image " + str(img) + "\n")
            
        score = score + loss_function(truth.loc[img].values, val, numcat)
        
    return -score/(numimg*numcat)
        
    
def loss_function(true, pred, ncat): 
    
    e = False
    eps = 1e-15
    imgscore = 0.0
    
    
    for i in range(0,ncat):
        if(true[i] == 1):
            e = True if pred[i] ==0 else False
            imgscore = imgscore + math.log(pred[i]+e*eps)
        else: 
            e = True if pred[i]==1 else False
            imgscore = imgscore + math.log(1-pred[i]+e*eps)
        
    return imgscore	    

if __name__ == "__main__":
    print(scoring (str(sys.argv[1]),str(sys.argv[2])))
    
