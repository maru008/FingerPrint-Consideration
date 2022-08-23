import pickle
import numpy as np
import pandas as pd

# with open("../data/node2fp_revised_1120.pickle",'rb') as f:
#     fingreprint_data_all = pickle.load(f)

with open("../data/hidden_vec_fingreprint.pickle",'rb') as f:
    hidden_fingreprint_data = pickle.load(f)

all_hidden_fingreprint_data = {}

node_data = pd.read_csv('../data/nodes_8212.csv')


for key_i in node_data["node_id"]:
    if key_i in hidden_fingreprint_data.keys():
        all_hidden_fingreprint_data[key_i] = hidden_fingreprint_data[key_i]
    else:
        all_hidden_fingreprint_data[key_i] = None
        
with open("../data/hidden_vec_fingreprint_{}.pickle".format(len(node_data)),'wb') as f:
     pickle.dump(all_hidden_fingreprint_data,f)