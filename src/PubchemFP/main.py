import pickle
import numpy as np
from gensim.models import word2vec
import logging



with open("../data/node2fp_8212.pickle", "rb") as f:
    FP_data = pickle.load(f)



id_ls_dict = {}
sentence_mtx = []


for node_id in FP_data.keys():
    val_i = FP_data[node_id]
    try:
        if val_i is not None:
            id_ls = []
            for i,FP_i in enumerate(val_i):
                if FP_i == 1:
                    add_id_str = f"id_{i}"
                    id_ls.append(add_id_str)
            sentence_mtx.append(id_ls)
            id_ls_dict[node_id] = id_ls
    except TypeError as e:
        # print(val_i,e)
        id_ls_dict[node_id] = val_i
        

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(sentences=sentence_mtx, vector_size=300, window=881, min_count=1)

# print(model.wv['id_1'])

id_vec_dict = {}


for node_id in id_ls_dict.keys():
    val_i = id_ls_dict[node_id]
    try:
        if val_i is not None:
            vec_i = np.zeros(300)
            for id_i in val_i:
                vec_i += model.wv[id_i]
            id_vec_dict[node_id] = vec_i
    except TypeError as e:
        id_vec_dict[node_id] = val_i


with open("../data/node2fp_word2vec_8212.pickle" , "wb") as f:
    pickle.dump(id_vec_dict,f)

