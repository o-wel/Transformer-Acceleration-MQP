import torch
import torch.nn as nn
from torch.nn import functional as F

class Transformer():
    
    def linear(self):

    def feedforward(self, numEmbeddings):

    def softmax(self,X):
        for i in range(len(X)):
            summation = sum(np.exp(X[i]))
            for j in range(len(X[i])):
                X[i][j] = np.exp(X[i][j])/summation



    def attention(self,Q,K,V):
        QK = (Q * K.T)/ndim
        return softmax(QK)*V

def forward():
    model = Transformer().to(device)


def setGPUInTorch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
