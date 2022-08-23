# 次元圧縮した画像を生成する
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_tSNE_fig(path):
    with open(path, mode='rb') as f:
        all_c_data = pickle.load(f)
    # compound_data  = {k: v for k, v in all_c_data.items() if (v != None).any()}
    print(len(all_c_data))
    compound_data = {}
    for key_i,val_i in zip(all_c_data.keys(),all_c_data.values()):
        if (val_i != None).any():
            compound_data[key_i] = val_i

    
    # compound_data  = {k: v for k, v in all_c_data.items() if (v != None).any()}
    
    tsne = TSNE(n_components=2, random_state=41)
    X_reduced = tsne.fit_transform(list(compound_data.values()))

    title = path.replace('.pickle','').replace('../data/','')
    m = 7
    n = 7
    plt.figure(figsize=(m, n))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1],s=15, alpha=0.5)
    plt.axis('off')
    plt.savefig('../fig/{}_{}_{}.jpg'.format(title,m,n))
    plt.show()
    
    
senkou_path = "../data/node2fp_revised_1120.pickle"
save_tSNE_fig(senkou_path)

# senkou_path = "../data/FlavorDB_train2vec.pickle"
# save_tSNE_fig(senkou_path)