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
from sklearn import svm
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

datadir = 'D:\\Data\\TalkingData'
eventCount = os.path.join(datadir,'event_count_per_device.json')
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), dtype={'device_id':str})
gatrain.set_index('device_id', inplace=True)

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

def distance(dict1, dict2): #Серенсена = mcc 0.132
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = 2*len(dict1.keys() & dict2.keys())
    d= a + b;
    return 1 if d == 0 else 1 - c/d;
 

def distance(dict1, dict2): #Шимкевича-Симпсона = mcc 0.16
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = len(dict1.keys() & dict2.keys())
    d= min(a, b);
    return 1 if d == 0 else 1 - c/d;

def readDict():
    with open(eventCount) as jsonFile:
        return json.load(jsonFile)

dict = readDict()

dictLen =1000#len(dict)
dictValues = list(dict.values());
distMatrix = np.zeros((dictLen, dictLen))
for i in range(dictLen):
    for j in range(i, dictLen):
        distMatrix[i,j] = distMatrix[j,i] = distance(dictValues[i], dictValues[j])
    print("{}/{}".format(i, dictLen))
np.save(os.path.join(datadir,'distanseMatrix'), distMatrix);
#distMatrix = np.load(os.path.join(datadir,'distanseMatrix.npy'))

#fastMapRes = FastMap(distMatrix).map(2)

print("MDS start")
MDS_metric = sklearn.manifold.MDS(metric = False, dissimilarity='precomputed')
fastMapRes = MDS_metric.fit_transform(distMatrix);
print("MDS end")
fig = plt.figure()
ax = fig.add_subplot(111)

colors = np.empty(dictLen, dtype=str)
for ix, deviceId in enumerate(list(dict.keys())[:dictLen]):
    if deviceId in gatrain.index:
        colors[ix] = 'r' if gatrain.loc[deviceId]['gender'] == 'F' else 'b'
    else:
        colors[ix] = 'w'
ax.scatter(fastMapRes[:,0],fastMapRes[:,1], c = colors)



#Radial basis function kernel
Y = [0 if gatrain.loc[deviceId]['gender'] == 'F' else 1 for deviceId in list(dict.keys())[:dictLen]];

print(Y)

svmkernel = np.zeros((dictLen, dictLen))
print("calc kernel")
for i in range(dictLen):
    for j in range(i, dictLen):
        svmkernel[i,j] = svmkernel[j, i] = math.exp(-2 * distMatrix[i,j])
clf = svm.SVC(kernel='precomputed')
print("fit svm")
clf.fit(svmkernel, Y) 
print("predict svm")
preds = clf.predict(svmkernel)

print("accuracy_score: {:.3f}".format(accuracy_score(Y, preds)));
print("roc_auc_score: {:.3f}".format(roc_auc_score(Y, preds)));
print("mcc: {:.3f}".format(matthews_corrcoef(Y, preds)));

plt.show()
        