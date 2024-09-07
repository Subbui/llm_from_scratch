import torch
import torch.nn as nn
from dataloader import PrepareData,dataload,dataset,input1

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
        super().__init__()
        assert d_out%num_heads==0,'output dimension should be divisible by number of heads'

        self.d_out=d_out
        self.context_length = context_length
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        # self.token_emb = nn.Embedding(100025,4)
        self.dropout=nn.Dropout(dropout)
        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj=nn.Linear(d_out,d_out)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
                             
    def forward(self,x):
        print(x.shape)
        batch,tokens,d_in = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value=self.W_value(x)
        
        key = key.view(batch,tokens,self.num_heads,self.head_dim)
        query = query.view(batch,tokens,self.num_heads,self.head_dim)
        value = value.view(batch,tokens,self.num_heads,self.head_dim)

        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        attn_scores = query @ key.transpose(2,3)
        mask_bool = self.mask.bool()[:tokens,:tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/key.shape[-1]**0.5,dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ value
        context_vec = context_vec.contiguous().view(batch,tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    

input1 = input1.type(torch.FloatTensor)
batch_data = torch.stack((input1,input1),dim=0)
print(batch_data.shape)
ca = MultiHeadAttention(d_in=10, d_out=3, context_length=10, dropout=0.1, num_heads=3)
logits=  ca(batch_data)
print(logits)

# inputs = torch.tensor(
# [[0.43, 0.15, 0.89], # Your (x^1)
# [0.55, 0.87, 0.66], # journey (x^2)
# [0.57, 0.85, 0.64], # starts (x^3)
# [0.22, 0.58, 0.33], # with (x^4)
# [0.77, 0.25, 0.10], # one (x^5)
# [0.05, 0.80, 0.55]] # step (x^6)
# )

# batch_input = torch.stack((inputs,inputs))
# batch_size, context_length, d_in = batch_input.shape
# d_out = 2
# mha= MultiHeadAttention(d_in=d_in,d_out=d_out,context_length=context_length,num_heads=2,dropout=0.1)
# context_vec = mha(batch_input)
# print(context_vec)
