from copyreg import pickle
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10  # CIFAR10もインポートしておく
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 300)
        # self.fc2 = torch.nn.Linear(512, 64)
        # self.fc3 = torch.nn.Linear(64, 16)
        # self.fc4 = torch.nn.Linear(16, 2)
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc1(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # self.fc1 = torch.nn.Linear(2, 16)
        # self.fc2 = torch.nn.Linear(16, 64)
        # self.fc3 = torch.nn.Linear(64, 512)
        self.fc4 = torch.nn.Linear(300, output_size)
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), axis=0)  # 0 or 1に変換
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, org_size):
        super().__init__()
        self.enc = Encoder(org_size)
        self.dec = Decoder(org_size)
    def forward(self, x):
        x = self.enc(x)  # エンコード
        hidden_layer = x
        x = self.dec(x)  # デコード
        return hidden_layer,x
    
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = MNIST('./data', train=True, transform=transform, download=True)
# testset = MNIST('./data', train=False, transform=transform, download=True)

with open("node2fp_revised_1120.pickle",'rb') as f:
    fingreprint_data_all = pickle.load(f)

print("all ingredient : ",len(fingreprint_data_all))
fingreprint_data  = {k: v for k, v in fingreprint_data_all.items() if (v != None).any()}

print("only compound : ",len(fingreprint_data))
batch_size = 100
# trainloader = DataLoader(fingreprint_data, batch_size=batch_size, shuffle=True)
trainloader = fingreprint_data
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

def train(net, criterion, optimizer, epochs, trainloader):
    '''
    net : モデルを入力 \n
    criterion : lossの定義
    optimizer : 
    '''
    losses = []
    output_and_label = []
    end_check = False
    hidden_dict = {}
    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0
        
        if epoch == EPOCHS:
            end_check = True

        for counter, trainloader_i in enumerate(zip(trainloader.keys(),trainloader.values())):
            key_i = trainloader_i[0]
            vec_i = trainloader_i[1]
            optimizer.zero_grad()
            # vec_i = vec_i.reshape(-1, input_size)
            vec_i = torch.tensor(vec_i).float()
            hidden,output = net(vec_i)
            
            loss = criterion(output, vec_i)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if end_check:
                hidden_dict[key_i] = hidden
            
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print('loss:', avg_loss)
        output_and_label.append((output, vec_i))
    print('finished')
    
    with open('hidden_vec_fingreprint.pickle','wb') as f:
        pickle.dump(hidden_dict,f)
    
    return output_and_label, losses

input_size = 881
net = AutoEncoder(input_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

EPOCHS = 150
output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)


plt.plot(losses)
plt.show()