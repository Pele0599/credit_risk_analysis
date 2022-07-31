import torch.nn as nn
from numpy import sqrt 
import torch

class NeuralNet(nn.Module):
    def __init__(self,input,layer_dim,output):
        super(NeuralNet,self).__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(input, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim,output)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self,input_data):
        x=self.dense_layers(input_data)
        predictions=self.softmax(x).squeeze(1)
        return predictions

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    losses = 0
    k = 0
    for inputs, targets in data_loader:
        inputs,targets = inputs.to(device).float(), targets.to(device).float()
        prediction = model(inputs)
        loss = loss_fn  (prediction, targets) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += nn.MSELoss()(prediction, targets)
        k += 1
    return losses / k


def train(model, data_loader, loss_fn, optimizer, epochs):
    loss = 0
    cont_loss = []  
    for _ in range(epochs):
        loss = train_epoch(model, data_loader, loss_fn, optimizer, 'cpu')
        cont_loss.append(sqrt(loss.detach().cpu().numpy()))
    return cont_loss

