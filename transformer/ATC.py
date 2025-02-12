import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=400, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        pe.to(device)
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))  / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        #print(q.shape, k.shape, v.shape)
        #print(mask.shape, scores.shape)
        #print(mask)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # batchsize, seq_len, # of heads, dimension of each head
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # 64       , -1,    , 8,        , 64
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model=d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads=heads, d_model=d_model)
        
        self.norm_2 = Norm(d_model=d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(q=x2, k=x2, v=x2, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, heads, dropout, N):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, src_mask):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)

        self.norm_2 = Norm(d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)

        self.norm_3 = Norm(d_model)
        self.dropout_3 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_output, src_mask, tgt_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(q=x2, k=x2, v=x2, mask = tgt_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(q=x2, k=e_output, v=e_output, mask=src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, heads, N, vocab_size = 9, dropout = 0.1):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model=d_model)
        self.layers = get_clones(DecoderLayer(d_model=d_model, heads=heads, dropout=dropout), N)
        self.norm = Norm(d_model)
    
    def forward(self, tgt, e_output, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_output, src_mask, tgt_mask)
        return self.norm(x)
    
class Model(nn.Module):
    def __init__(self, d_model=512, tgt_size=9, heads = 8, dropout=0.1, N = 6):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, heads=heads, dropout=dropout, N=N)
        self.decoder = Decoder(d_model=d_model, heads=heads, dropout=dropout, N=N)
        self.out = nn.Linear(d_model, tgt_size)

    def forward(self, input, tgt, mask_input, mask_tgt):
        e_outputs = self.encoder(input, mask_input)
        d_outputs = self.decoder(tgt, e_outputs, mask_input, mask_tgt)
        output = self.out(d_outputs)
        return output