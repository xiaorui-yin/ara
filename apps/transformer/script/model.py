# The APACHE License (APACHE)

# Copyright (c) 2022 Xiaorui Yin. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        
        return x

net1 = Net(1,10,1)
optimizer = torch.optim.SGD(net1.parameters(),lr=0.1)
loss_func = nn.MSELoss()


for i in range(100):
    prediction = net1(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

torch.save(net1,'net.pkl')
torch.save(net1.state_dict(),'net_parms.pkl')


net2 = torch.load('net.pkl')
net3 = Net(1,10,1)
net3.load_state_dict(torch.load('net_parms.pkl'))

parm = {}
for name,parameters in net3.named_parameters():
    print(name,':',parameters.size())
    parm[name] = parameters.detach().numpy()

w1 = parm['hidden.weight']
b1 = parm['hidden.bias']
w2 = parm['predict.weight']
b2 = parm['predict.bias']
print(w1)
print(b1)
print(w2)
print(b2)
# print('size:',w1.shape[0])
f = open('./w1.txt',mode = 'w+')
f.write('const float w1[%d] = {\n'% w1.shape[0]*w1.shape[1])
t = str(w1[1,0])
print(t)
for j in range(w1.shape[0]):
    for i in range(w1.shape[1]):
        f.write(str(w1[j,i])+',\n')
        pass
    pass
f.write('}\n')
f.close()
