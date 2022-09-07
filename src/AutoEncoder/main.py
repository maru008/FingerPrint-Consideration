from AutoEncoder import My_Module
from AutoEncoder import train
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd

input_size = 881
net = My_Module(input_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

EPOCHS = 100
hidden_fingreprint_data,output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)




node_data = pd.read_csv('../../data/nodes_8212.csv')

all_hidden_fingreprint_data = {}

for key_i in node_data["node_id"]:
    if key_i in hidden_fingreprint_data.keys():
        all_hidden_fingreprint_data[key_i] = hidden_fingreprint_data[key_i]
    else:
        all_hidden_fingreprint_data[key_i] = None
        
with open("../../data/hidden_vec_fingreprint_300.pickle",'wb') as f:
     pickle.dump(all_hidden_fingreprint_data,f)
     
plt.plot(losses)
plt.show()