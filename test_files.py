from model import *
from final import *



def test_cnn():
    cnn = CNN(cnn_emb_size)
    i = 0
    for image, _ in train_dl:
        i += 1
        print(cnn(image).size())
        if i == 10:
            break

def test_rnn():
    features = torch.rand((1, cnn_emb_size))
    c = "<SOS> a giant red skateboard with wheels <EOS>"
    t_caption = list(map(lambda x: vocab2idx.get(x, 3), c.split(' ')))
    rnn = RNN(rnn_emb_size, hidden_size, vocab_len, batch_size)
    output = rnn(features, torch.tensor(t_caption).unsqueeze(0).long())

    print("output size", output.size())
    print("caption size", torch.tensor(t_caption).unsqueeze(0).long().size())


def test_full():
    full = FullModel(cnn_emb_size, rnn_emb_size, vocab_len, batch_size)
    i = 0
    for image, caption in train_dl:
        i += 1
        print('output sizes')
        print(full(image, caption).size())
        if i == 5:
            break