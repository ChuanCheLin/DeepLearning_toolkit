from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scatter', action='store_true', help='draw the scatter plot instead of the default kde plot')
args = parser.parse_args()  

path = "/home/eric/few-shot-object-detection/features.npy"
label_path = "/home/eric/few-shot-object-detection/labels.npy"
features = np.load(path) # 1024 x N
labels = np.load(label_path) # 1 x N

table = ['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 
'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 
'miner', 'thrips', 'tetrany', 'formosa', 'other']

# tsne
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(features)

joblib.dump(X_tsne, 'tsne.pkl')

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
plt.figure(figsize=(10, 10))

for i in range(len(table)):
    x = []
    y = []
    # set color
    if i in range(0, 11):
            color = plt.cm.Set3(i)
    elif i in range(11, 17):
            color = plt.cm.Set1(i-10)

    #
    for j in range(X_norm.shape[0]):
        if labels[j] == i:
            x.append(X_norm[j, 0])
            y.append(X_norm[j, 1])

    if args.scatter:
        plt.scatter(x, y, color = color, label = table[i])
    else:
        sns.kdeplot(x, y, color = color, label = table[i], levels=15)


plt.xticks([])
plt.yticks([])
plt.legend()
plt.savefig('t_sne.png')
plt.show()


    # plt.text(X_norm[i, 0], X_norm[i, 1], '.', color = color, #str(labels[i])
    #         fontdict={'weight': 'bold', 'size': 9})