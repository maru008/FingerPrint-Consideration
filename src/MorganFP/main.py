import pickle
import pandas as pd


with open("../../data/node2fp_8212.pickle", "rb") as f:
    FP_data = pickle.load(f)
    
#MorganFPデータの取得
f = open('../../data/FlavorDBmols.cp_UNK', 'r')

chemi_node_data = pd.read_csv("../../data/FlavorDB2vec.csv")
chemi_node_id_data = chemi_node_data[["nodeID","ID"]]

datalist = f.readlines()

MorganFP_ls = []
all_MorganFP_ls = []

for dataL in datalist:
    data_ls = dataL.split()
    MorganFP_ls.append(data_ls)
    all_MorganFP_ls.extend(data_ls)
    
chemi_node_id_data["Morgan_id_ls"] = MorganFP_ls

set_all_MorganFP_ls = list(set(all_MorganFP_ls))
dim = len(set_all_MorganFP_ls)
print("全次元数:",dim)
# print(set_all_MorganFP_ls)

node_data = pd.read_csv("../../data/nodes_8212.csv")
node_ls = node_data["node_id"]
chemical_node_id_ls = node_data[node_data["node_type"] == 'compound']["node_id"].values

Morgan2binary_dict = {}

for node_id in FP_data.keys():
    # print(node_id)
    if node_id in chemical_node_id_ls:
        #化合物に対する処理
        trg_Morgan_ls = list(chemi_node_id_data[chemi_node_id_data["nodeID"] == node_id]["Morgan_id_ls"].values[0])
        binary_ls = []
        for id_i in set_all_MorganFP_ls:
            if id_i in trg_Morgan_ls:
                binary_ls.append(1)
            else:
                binary_ls.append(0)
                # print("0")
        Morgan2binary_dict[node_id] = binary_ls   
    else:
        Morgan2binary_dict[node_id] = FP_data[node_id]

with open(f"../../data/Morgan2binary_{dim}.pickle","wb") as f:
    pickle.dump(Morgan2binary_dict,f)
    