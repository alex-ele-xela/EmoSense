import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class SpeechEmotionRecog(nn.Module):
    def __init__(self, num_labels=8):
        super(SpeechEmotionRecog, self).__init__()
        
        FRAME_LENGTH = 2014
        HOP_LENGTH = 512

        self.batch_norm = nn.BatchNorm1d(1)

        self.frame = nn.Conv1d(in_channels=1,
                               out_channels=16,
                               kernel_size=FRAME_LENGTH,
                               stride=HOP_LENGTH)

        self.conv1 = nn.Conv1d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)

        self.batch_norm1 = nn.BatchNorm1d(32)

        self.pe = self.getPositionEncoding(212, 32)

        self.encoder = nn.TransformerEncoderLayer(d_model=32,
                                                  nhead=2,
                                                  dropout=0.2,
                                                  activation='gelu',
                                                  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder,
                                                         num_layers=1)
        
        self.batch_norm2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.batch_norm3 = nn.BatchNorm1d(32)
        
        self.linear1 = nn.Linear(in_features=6784,
                                 out_features=1024)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.logits = nn.Linear(in_features=1024,
                                out_features=num_labels)
        
    def getPositionEncoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return torch.tensor(P).type(torch.float).squeeze()


    def forward(self, audios):
        norm_audio = self.batch_norm(audios)
        frames = self.frame(norm_audio)

        conv1 = self.conv1(frames)
        conv1 = self.gelu(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.gelu(conv2)
        conv2 = self.batch_norm1(conv2)

        conv2 = torch.transpose(conv2, dim0=1, dim1=2)

        pencoded = conv2 + self.pe.to(conv1.device)

        encoded = self.transformer_encoder(pencoded)
        encoded = torch.transpose(encoded, dim0=1, dim1=2)
        encoded = self.batch_norm2(encoded)

        conv3 = self.conv3(encoded)
        conv3 = self.gelu(conv3)
        conv3 = self.batch_norm3(conv3)

        flattened = torch.flatten(conv3, start_dim=1, end_dim=-1)
        lin1 = self.linear1(flattened)
        lin1 = self.gelu(lin1)
        lin1 = self.dropout1(lin1)

        logits = self.logits(lin1)

        return logits
