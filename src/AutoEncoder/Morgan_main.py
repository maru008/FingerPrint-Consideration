import pickle
import torch
from My_Module import AutoEncoder,train
import matplotlib.pyplot as plt
import pandas as pd

with open('../../data/Morgan2binary_445.pickle',"rb") as f:
    MorganBinary_data = pickle.load(f)

fingreprint_data  = {k: v for k, v in MorganBinary_data.items() if (v != None)}

print("only compound : ",len(fingreprint_data))
batch_size = 100
trainloader = fingreprint_data

input_size = 445
net = AutoEncoder(input_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

EPOCHS = 100

hidden_Morgan_FP,output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)

node_data = pd.read_csv('../../data/nodes_8212.csv')

all_hidden_Morgan_FP = {}

for key_i in node_data["node_id"]:
    if key_i in hidden_Morgan_FP.keys():
        all_hidden_Morgan_FP [key_i] = hidden_Morgan_FP[key_i]
    else:
        all_hidden_Morgan_FP [key_i] = None
        
with open("../../data/hidden_vec_fingreprint_100.pickle",'wb') as f:
     pickle.dump(all_hidden_Morgan_FP ,f)
     
plt.plot(losses)
plt.show()