from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import joblib

label_path = "/home/eric/few-shot-object-detection/labels.npy"
labels = np.load(label_path)

table = ['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 
'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 
'miner', 'thrips', 'tetrany', 'formosa', 'other']

# list for selection
chosen = [5, 6, 7, 8, 9] # roller to caloptilia

# tsne
X_tsne = loaded_model = joblib.load('tsne_3d.pkl')

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
fig = plt.figure()
ax = Axes3D(fig)

#for i in range(len(table)):
for i in chosen:
    x = []
    y = []
    z = []
    # set color
    if i in range(0, 11):
            color = plt.cm.Set3(i)
    elif i in range(11, 17):
            color = plt.cm.Set1(i-10)

    for j in range(X_norm.shape[0]):
        if labels[j] == i:
            x.append(X_norm[j, 0])
            y.append(X_norm[j, 1])
            z.append(X_norm[j, 2])

    ax.scatter(x, y, z, c = color)

plt.axis('off')
plt.show()