import json
import os
import numpy as np
import pandas as pd
from FastMap import FastMap as FastMap
from scipy.spatial.distance import  pdist, squareform
import matplotlib.pyplot as plt
import sklearn
import sklearn.manifold
import math
import time
from sklearn import svm
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

datadir = 'D:\\Data\\TalkingData'
eventCount = os.path.join(datadir,'event_count_per_device.json')
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), dtype={'device_id':str})
gatrain.set_index('device_id', inplace=True)


def log(func):
    def logWrapper(*args, **kwargs):
        print("Start evaluate '{}' function".format(func.__name__))
        start = time.time();
        res = func(*args, **kwargs)
        end = time.time();
        print("End evaluate '{}' function ({} s) ".format(func.__name__, end-start))
        return res;
    return logWrapper

def cache(fromCache=True):
    def cacheDecorator(func):
        def cacheWrapper(*args, **kwargs):
            fileName = os.path.join(datadir, func.__name__ + ".npy")
            if fromCache & os.path.isfile(fileName):
                print("Get {} from cache".format(func.__name__))
                return np.load(fileName);
            else:
                print("Start evaluate '{}' function".format(func.__name__))
                start = time.time();
                result = func(*args, **kwargs)
                end = time.time();
                print("End evaluate '{}' function ({} s) ".format(func.__name__, end-start))
                np.save(fileName, result);
                return result;
        return cacheWrapper
    return cacheDecorator

#def distance(dict1, dict2): #(Jaccard) = mmc = 0
#    a = len(dict1.keys())
#    b = len(dict2.keys())
#    c = len(dict1.keys() & dict2.keys())
#    d= (a+b-c);
#    return 1 if d == 0 else 1 - c/d;

#def distance(dict1, dict2): #Шимкевича-Симпсона = mcc 0.16
#    a = len(dict1.keys())
#    b = len(dict2.keys())
#    c = len(dict1.keys() & dict2.keys())
#    d= min(a, b);
#    return 1 if d == 0 else 1 - c/d;

#def distance(dict1, dict2): #Серенсена = mcc 0.132
#    a = len(dict1.keys())
#    b = len(dict2.keys())
#    c = 2*len(dict1.keys() & dict2.keys())
#    d= a + b;
#    return 1 if d == 0 else 1 - c/d;
 

def distance(dict1, dict2): #Шимкевича-Симпсона = mcc 0.16
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = len(dict1.keys() & dict2.keys())
    d= min(a, b);
    return 1 if d == 0 else 1 - c/d;

def readDict():
    with open(eventCount) as jsonFile:
        return json.load(jsonFile)

@cache(fromCache=True)
def calcDistMatrix(dictValues, dictLen):
    distMatrix = np.zeros((dictLen, dictLen))
    for i in range(dictLen):
        for j in range(i, dictLen):
            distMatrix[i,j] = distMatrix[j,i] = distance(dictValues[i], dictValues[j])
    return distMatrix;

@log
def scale(distMatrix):
    #return FastMap(distMatrix).map(2)
    MDS_metric = sklearn.manifold.MDS(metric = False, dissimilarity='precomputed')
    return MDS_metric.fit_transform(distMatrix);


def plot2D(gatrain, scaled):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = np.empty(scaled.shape[0], dtype=str)
    for ix, deviceId in enumerate(list(dict.keys())[:dictLen]):
        if deviceId in gatrain.index:
            colors[ix] = 'r' if gatrain.loc[deviceId]['gender'] == 'F' else 'b'
        else:
            colors[ix] = 'w'
    ax.scatter(scaled[:,0],scaled[:,1], c = colors)

@log
def predict(X, Y):
    size = X.shape[0]
    svmkernel = np.zeros((size, size))
    print("calc kernel")
    for i in range(size):
        for j in range(i, size):
            svmkernel[i,j] = svmkernel[j, i] = math.exp(-2 * X[i,j])
    clf = svm.SVC(kernel='precomputed')
    clf.fit(svmkernel, Y) 
    return clf.predict(svmkernel)

dict = readDict()
dictLen =1000#len(dict)
dictValues = list(dict.values());
distMatrix = calcDistMatrix(dictValues, dictLen)
scaled = scale(distMatrix)

plot2D(gatrain, scaled)

Y = [0 if gatrain.loc[deviceId]['gender'] == 'F' else 1 for deviceId in list(dict.keys())[:dictLen]];
preds = predict(distMatrix, Y)

print("accuracy_score: {:.3f}".format(accuracy_score(Y, preds)));
print("roc_auc_score: {:.3f}".format(roc_auc_score(Y, preds)));
print("mcc: {:.3f}".format(matthews_corrcoef(Y, preds)));

plt.show()
        