from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scatter', action='store_true', help='draw the scatter plot instead of the default kde plot')
args = parser.parse_args()        

label_path = "/home/eric/few-shot-object-detection/labels.npy"
labels = np.load(label_path)

table = ['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 
'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 
'miner', 'thrips', 'tetrany', 'formosa', 'other']

# list for selection
# chosen = [5, 6, 7, 8, 9] # roller to caloptilia
# chosen = [10, 11] # mosquito (stage)
# chosen = [6, 13] # moth, thrips
# chosen = [6, 7] # moth, tortrix
chosen = [7, 8] # tortrix, flushworm
# chosen = [3, 14] # sunburn, tetrany
# chosen = [0, 1, 2, 3, 4, 7, 9, 12] # excellent-perfomance mAP>80

# tsne
X_tsne = loaded_model = joblib.load('tsne.pkl')

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
plt.figure(figsize=(10, 10))

#for i in range(len(table)):
for i in chosen:
    x = []
    y = []
    # set color
    if i in range(0, 11):
            color = plt.cm.Set3(i)
    elif i in range(11, 17):
            color = plt.cm.Set1(i-10)

    for j in range(X_norm.shape[0]):
        if labels[j] == i:
            x.append(X_norm[j, 0])
            y.append(X_norm[j, 1])
    
    if args.scatter:
        plt.scatter(x, y, color = color, label = table[i])
    else:
        sns.kdeplot(x, y, color = color, label = table[i], levels=15)
'''
for i in range(X_norm.shape[0]):
    #set color
    if labels[i] in range(0, 11):
        color = plt.cm.Set3(labels[i])
    elif labels[i] in range(11, 17):
        color = plt.cm.Set1(labels[i]-10)
    plt.scatter(X_norm[i, 0], X_norm[i, 1], color = color, label = table[labels[i]])
'''

plt.xticks([])
plt.yticks([])
plt.legend()
plt.savefig('t_sne.png')
plt.show()