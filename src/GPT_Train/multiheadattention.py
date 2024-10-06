import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out, num_heads,context_length,dropout=0.1,qkv_bias=False):
        super().__init__()
        assert d_out%num_heads==0,"output dim should be divisible by num_heads"
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        self.outproj = nn.Linear(d_out,d_out)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))


    def forward(self,x):
        num_batch,tokens,d_in = x.shape
        key = self.w_key(x)
        query = self.w_query(x)
        value = self.w_value(x)

        key = key.view(num_batch,tokens,self.num_heads,self.head_dim).transpose(1,2)
        query = query.view(num_batch,tokens,self.num_heads,self.head_dim).transpose(1,2)
        value = value.view(num_batch,tokens,self.num_heads,self.head_dim).transpose(1,2)

        attn_scores = query @ key.transpose(2,3)
        attn_mask = self.mask.bool()[:tokens,:tokens]
        attn_scores.masked_fill_(attn_mask,-torch.inf)

        attn_wts = torch.softmax(attn_scores/key.shape[-1]**0.5,dim=-1)
        attn_wts = self.dropout(attn_wts)

        context_vec = (attn_wts @ value).transpose(1,2)
        context_vec = context_vec.contiguous().view(num_batch,tokens,self.d_out)
        context_vec = self.outproj(context_vec)
        return context_vec
    

# inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
# [40, 1107, 588]]) # "I really like"]

# targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
# [107, 588, 11311]]) # " really like chocolate"]
# inputs = torch.tensor(
# [[0.43, 0.15, 0.89], # Your (x^1)
# [0.55, 0.87, 0.66], # journey (x^2)
# [0.57, 0.85, 0.64], # starts (x^3)
# [0.22, 0.58, 0.33], # with (x^4)
# [0.77, 0.25, 0.10], # one (x^5)
# [0.05, 0.80, 0.55]] # step (x^6)
# )
# batch = torch.stack((inputs,inputs),dim=0)
# # print(batch.shape)
# torch.manual_seed(123)
# batch_size, context_length, d_in = batch.shape
# d_out = 2
# mha = MultiHeadAttention(d_in, d_out, context_length=context_length,dropout= 0.0, num_heads=2)
# context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)