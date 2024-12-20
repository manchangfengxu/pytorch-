import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append("..")

import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use', device)

lr = 1e-2
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens, num_layers=2, bidirectional=True, dropout=0.5)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)