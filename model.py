import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision.models as models

import torchvision.transforms as T
from PIL import Image

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class FlickrDataset(Dataset):
    def __init__(self, df, img_dir_path, vocab=None, max_length=50, **kwargs):
        self.img_dir_path = img_dir_path
        self.vocab = vocab
        self.max_len = max_length
        self.filename = df.filename.values

        self.caption = df.cleaned.apply(lambda x: self.preprocess(x)).values
    
    def preprocess(self, text):
        text = ['<SOS>'] + text + ['<EOS>']
        text_len = len(text)
        start = max(0, self.max_len - text_len)  

        x = torch.zeros(self.max_len)
        for i in range(min(text_len, self.max_len)):
            x[i+start] = self.vocab.get(text[i], 3)

        return x
    
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, idx):
        image_path = f"{self.img_dir_path}/{self.filename[idx]}"
        image = Image.open(image_path).convert('RGB')

        image = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
            ])(image)
        caption = torch.tensor(self.caption[idx]).long()
        return image, caption


class CNN(nn.Module):
    def __init__(self, emb_size):
        super(CNN, self).__init__()
        model = models.resnet18(pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, emb_size).to(device)
        self.model = model
        
    def forward(self, image):
        features = self.model(image)
        return features

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, batch_size=32):
        super(RNN, self).__init__()
        """
        Input size matches embedding size since the inputs are embeddings.
        Output size is vocab_len since we are generating words. 
        Hidden size is LSTM hidden layer size. 
        """
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.emb = nn.Embedding(num_embeddings=vocab_len, 
                                      embedding_dim=emb_size).to(device)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, vocab_len).to(device)
        self.vocab_len = vocab_len

    def init_hidden(self, features):
        hidden = torch.autograd.Variable(
            torch.zeros(1, self.batch_size, self.hidden_size)).to(device)
        cell = torch.autograd.Variable(features.unsqueeze(0)).to(device)
        return (hidden, cell)
        
    def forward(self, features, captions):
        h_state, c_state = self.init_hidden(features)
        embed = self.emb(captions)
        lstm_out, (h_state, c_state) = self.lstm(embed, (h_state, c_state))

        outputs = self.fc(lstm_out)
        outputs = outputs.view(-1, self.vocab_len)
        
        return outputs


class FullModel(nn.Module):
    def __init__(self, cnn_emb_size, rnn_emb_size, vocab_size, batch_size):
        super(FullModel, self).__init__()
        self.cnn = CNN(cnn_emb_size).to(device)
        self.rnn = RNN(rnn_emb_size, cnn_emb_size, vocab_size, batch_size).to(device)

    def forward(self, images, captions):
        features = self.cnn(images)
        outputs = self.rnn(features, captions)
        return outputs


