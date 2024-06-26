import torch
from torch import nn

debug = False


def print_msg(*msg):
    if debug:
        print(msg)
"""
class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h,mask):
        u_it = self.u(h)
        u_it = self.tanh(u_it)
       
        alpha = torch.exp(torch.matmul(u_it, self.v))
        alpha = mask * alpha + self.epsilon
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        alpha = mask * (alpha / denominator_sum)
        #Batch matrix multiplication
        output = h * alpha.unsqueeze(2)
        output = torch.sum(output, dim=1)
        #print_msg('output ', output.shape)

        return output, alpha
 """
class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, mask):
       u_it = self.u(h)
       u_it = self.tanh(u_it)
       
       alpha = torch.exp(torch.matmul(u_it, self.v))
       alpha = mask * alpha + self.epsilon
       denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
       alpha = mask * (alpha / denominator_sum)
       output = h * alpha.unsqueeze(2)
       output = torch.sum(output, dim=1)
       return output, alpha


class DoubleAttention(nn.Module):
    def __init__(self, dimension,dimension1,dimension2):
        super(DoubleAttention, self).__init__()

        self.u = nn.Linear(dimension1, dimension)
        self.z = nn.Linear(dimension2, dimension)
        self.m = nn.Linear(2*dimension, dimension1)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h1,h2):
        u_it = self.u(h1)
        z_it = self.z(h2)
        u_it = torch.cat([h1,h2], dim=-1)
        u_it = self.m(u_it)
        u_it = self.tanh(u_it)
        """ Direct softmax considers whole sequence. It contains padding. So manual"""
        alpha = torch.exp(torch.matmul(u_it, self.v))
        alpha = mask * alpha + self.epsilon
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        alpha = mask * (alpha / denominator_sum)
        #Batch matrix multiplication
        output = h * alpha #.unsqueeze(2)
        output = torch.sum(output, dim=1)
        #print_msg('output ', output.shape)

        return output, alpha


class EmotionAttention(nn.Module):
    def __init__(self, dimension):
        super(EmotionAttention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.g = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, e, mask):


        u_it = self.u(h) + self.g(e)
        u_it = self.tanh(u_it)

        alpha = torch.exp(torch.matmul(u_it, self.v))
        alpha = mask * alpha + self.epsilon
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        alpha = mask * (alpha / denominator_sum)
        output = h * alpha.unsqueeze(2)
        output = torch.sum(output, dim=1)

        return output, alpha



class BaseEmotionAttention(nn.Module):
    def __init__(self, dimension):
        super(BaseEmotionAttention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.g = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, e, mask):
        print_msg(mask)

        e_attn = torch.unsqueeze(self.g(e),1)
        # b, t, dim = h.size()
        # h : Batch * timestep * dimension
        print_msg('h', h.shape)

        # u(h) : Batch * timestep * att_dim
        u_it = self.u(h) + e_attn
        print_msg('u(h)', u_it.size())

        # tan(x) : Batch * timestep * att_dim
        u_it = self.tanh(u_it)

        # alpha = self.softmax(torch.matmul(u_it, self.v))
        # print_msg(alpha)
        # alpha = mask * alpha

        """ Direct softmax considers whole sequence. It contains padding. So manual"""
        # # softmax(x) : Batch * timestep * att_dim
        alpha = torch.exp(torch.matmul(u_it, self.v))
        print_msg(alpha)

        alpha = mask * alpha + self.epsilon
        # print_msg('after mask', alpha, alpha.size())
        #
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        print_msg(denominator_sum, denominator_sum.size())
        #
        alpha = mask * (alpha / denominator_sum)

        # Batch matrix multiplication
        output = h * alpha.unsqueeze(2)
        # print_msg('output ', output.shape)

        output = torch.sum(output, dim=1)
        # print_msg('output ', output.shape)

        return output, alpha


if __name__ == '__main__':
    # debug = True

    """ Test """
    data = [
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 0, 3, 2, 1, 2, 4],
            [0, 0, 3, 2, 3, 2, 1, 2, 4]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 4, 2, 2, 1],
         [0, 0, 0, 0, 0, 0, 1, 2, 3],
         [0, 0, 0, 0, 3, 2, 1, 2, 4],
         [0, 0, 3, 2, 3, 2, 1, 2, 4], ]
    ]

    data = torch.LongTensor(data)
    print(data)
    print(data.size())

    mask = (data > 0).float()
    print(mask)

    doc_mask = torch.sum(mask, dim=-1)
    print_msg(doc_mask)
    doc_mask = (doc_mask > 0).float()
    print_msg(doc_mask)

    emb = torch.nn.Embedding(5, 3)
    lstm = torch.nn.LSTM(3, 4)
    attn = Attention(4)

    x = emb(data)
    print(x.size())

    x, _ = lstm(x.view(-1, 5, 3))
    print(x.size())

    x, attn_weights = attn(x, mask)
    print(x.size(), attn_weights.size())
    print(attn_weights)
