import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size = 2
max_seq_len = 3
rnn_input_dim = 4
hidden_size = 2
n_layers =1

# container
batch_in = torch.zeros((batch_size, max_seq_len, rnn_input_dim))

#data
vec_1 = torch.FloatTensor([[1, 2, 3, 4], [1, 1, 1, 1], [2, 2, 2, 2]])
vec_2 = torch.FloatTensor([[1, 2, 1, 2], [1, 1, 1, 1], [0, 0, 0, 0]])

batch_in[0] = vec_1
batch_in[1] = vec_2

batch_in = Variable(batch_in)

print(batch_in)

seq_lengths = [3, 2] # list of integers holding information about the batch size at each sequence step

# pack it
pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)

print(pack)

# initialize
rnn = nn.RNN(rnn_input_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
h0 = Variable(torch.randn(n_layers * 2, batch_size, hidden_size))

#forward
out, _ = rnn(pack, h0)
print(out)

# unpack
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

print(unpacked)

linear = nn.Linear(2 * hidden_size, 5)

linear_out = linear(unpacked)
print(linear_out)


softmax = nn.Softmax()

softmax_out = torch.stack([softmax(linear_out[i]) for i in range(batch_size)], 0)
print(softmax_out)

# print(linear_out.size())
# linear_out = linear_out.resize(1, *linear_out.size())
# linear_out = linear_out.transpose(1, 3)
# print(linear_out)
# softmax_out = softmax2d(linear_out)
# softmax_out = softmax_out.transpose(1, 3)[0]
# print(softmax_out)


