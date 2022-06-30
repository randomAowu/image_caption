import os
import spacy
from collections import Counter
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'senter', 'tok2vec'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_df(CAPTIONS_PATH, IMG_DIR_PATH, sample=True):
    """
    Download and then preprocess texts from the caption dataset. 
    Extract the filename, caption, and caption version from the text. 
    If the sample boolean is True then create a subset according to the existing images.
    """
    with open(CAPTIONS_PATH) as f:
        texts = f.readlines()

    df = pd.DataFrame({'texts':texts})
    df['filename'] = df.texts.str.extract(r"(.*\.jpg)")
    df['caption'] = df.texts.str.extract(r"\t(.*).\n")
    df['version'] = df.texts.str.extract(r"\#(\d)")

    if sample:
        filenames = os.listdir(IMG_DIR_PATH)
        small = pd.merge(df, pd.Series(filenames, name='existing'), 
                        how='inner', 
                        left_on='filename', 
                        right_on='existing')
        
    return small if sample else df


def train_test_split(df, ratio=0.7):
    mask = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
    train_df = df.iloc[mask[:int(df.shape[0]*ratio)], :]
    valid_df = df.iloc[mask[int(df.shape[0]*ratio):], :]
    return train_df, valid_df


def generate_vocabulary(series, threshold=4):
    """
    Takes in a pandas series datatype and creates two vocabulary sets. 
    Returns a cleaned version of the text using the tokenizer from Spacy.
    """
    vocab2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx2vocab = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    ret_series = []
    count_dict = Counter()
    for row in series:
        tokenized_list = [token.text.lower() for token in nlp.tokenizer(row)]
        ret_series.append(tokenized_list)
        count_dict.update(tokenized_list)
    word_dict = set()

    for word in count_dict:
        if count_dict[word] > threshold:
            word_dict.add(word)

    for i, word in enumerate(word_dict, start=4):
        vocab2idx[word] = i
        idx2vocab[i] = word

    return vocab2idx, idx2vocab, ret_series


def func(model, optimizer, dl, update=True, vocab_len=1, batch_size=5):
    """
    Training and evaluation function for the full model.
    Reshaping and ignore index are needed for compatability issues.
    Batch size 1 does not work with this set up with Pytorch.
    """
    model.train() if update else model.eval()
    samples, total_loss = 0, 0

    for images, captions in tqdm(dl):
        images = images.to(device)
        captions = captions.to(device)
        outputs = model(images, captions)
        loss = F.cross_entropy(outputs.view(-1, vocab_len), captions.view(-1), ignore_index=0)
        total_loss += loss.item() * batch_size
        samples += batch_size

        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    avg_loss = total_loss/samples
    return avg_loss


def make_prediction(model, image_path):
    """
    Using a temporary local model, make a text prediction for the image. 
    The shapes are all different from the existing model, so the pipeline is rebuilt here.
    Iteration limit is at 50 since the model was trained with that as a maximum.
    """
    model = model.to('cpu')
    img = Image.open(image_path).convert('RGB')
    img = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
                ])(img)
    plt.imshow(img.transpose(2, 0).transpose(1, 0))
    
    model.eval()
    features = model.cnn(img.unsqueeze(0))
    output_ind = []
    state = (torch.zeros((1, model.rnn.hidden_size)), \
                torch.autograd.Variable(features))

    emb = model.rnn.emb(torch.tensor([1])) # <SOS>

    for i in range(50):
        lstm_out, state = model.rnn.lstm(emb, state)
        outputs = model.rnn.fc(lstm_out).view(-1, model.rnn.vocab_len)
        ind = np.argmax(outputs.detach().numpy())
        output_ind.append(ind) 
        
        if (ind == 2):
            break

        emb = model.rnn.emb(torch.tensor([ind]))

    print(" ".join([idx2vocab.get(idx) for idx in output_ind]))



# Set of temporary functions to translate captions to tokens and tokens to captions.
def func1(caption):
    print([idx2vocab.get(idx) for idx in caption.view(-1).numpy()])

def func2(outputs):
    c = np.argmax(outputs.view(-1, vocab_len).detach().numpy(), axis=1)
    print([idx2vocab.get(idx) for idx in c])

def func3(outputs, caption):
    func1(caption)
    func2(outputs)