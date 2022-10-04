from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
from sklearn.decomposition import PCA



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scatter', action='store_true', help='draw the scatter plot instead of the default kde plot')
args = parser.parse_args()  

path_root = "/home/eric/FSCE_tea-diseases/"
path = path_root + "features.npy"
label_path = path_root + "labels.npy"
features = np.load(path) # 1024 x N for RoI features, 128 x N for contrasitve features
labels = np.load(label_path) # 1 x N

table = ['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 
'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 
'miner', 'thrips', 'tetrany', 'formosa', 'other']

chosen = [3, 9, 14, 15]

# select limited samples for each class
limit = 30
# for i in range(len(table)):
for i in chosen:
    count = 0
    for j in range(len(labels)):
        if labels[j] == i and count < limit:
            features_temp = features[j]
            labels_temp = labels[j]
            if i == 3 and count == 0:
                features_new = features_temp
                labels_new = labels_temp
            else:
                features_new = np.vstack((features_new, features_temp))
                labels_new = np.vstack((labels_new, labels_temp))
            count += 1

# # pca 
# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(features_new)
# print("transformed shape:", X_pca.shape)

# tsne
X_tsne = manifold.TSNE(n_components=2, perplexity = 30, init='pca', random_state=5, verbose=1, n_iter=5000).fit_transform(features_new)

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
        if labels_new[j] == i:
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