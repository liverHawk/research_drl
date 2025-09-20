import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFlowNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, PORT_DIM=8, PROTOCOL_DIM=4):
        super(DeepFlowNetwork, self).__init__()
        self.protocol_embedding = nn.Embedding(256, PROTOCOL_DIM)
        self.port_embedding = nn.Embedding(65536, PORT_DIM)
        n_inputs = n_inputs - 2 + PORT_DIM + PROTOCOL_DIM # - 2 + protocol + port
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        port_emb = self.port_embedding(x[0].long())
        protocol_emb = self.protocol_embedding(x[1].long())
        renew = torch.cat([port_emb, protocol_emb, x[2]], dim=1)
        renew = F.relu(self.fc1(renew))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)