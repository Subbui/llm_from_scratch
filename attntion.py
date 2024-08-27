import torch
import torch.nn as nn
from dataloader import PrepareData,dataload,dataset,input1

class CasualAttention(nn.Module):
    def __init__(self,d_in,d_out,emb_dim,num_heads,dropout,qkv_bias=False):
        super().__init__()
        assert d_out%num_heads==0,'output dimension should be divisible by number of heads'

        self.d_out=d_out
        self.num_heads=num_heads
        self.emb_dim=emb_dim
        self.head_dim=d_out/num_heads
        self.dropout=nn.Dropout(dropout)
        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.register_buffer("mask",torch.ones(d_out,d_out))

    def forward(self,x):
        batch,tokens,d_out = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value=self.W_value(x)

        query = query.view(batch,tokens,self.num_heads,self.emb_dim)
        key = key.view(batch,tokens,self.num_heads,self.emb_dim)
        value = value.view(batch,tokens,self.num_heads,self.emb_dim)

        query = query.transpose(1,2)
        key = query.transpose(1,2)

        attn_scores = query @ key.transpose(2,3)
        attn_scores=self.dropout(attn_scores)
        attn_weights = torch.softmax(attn_scores/key.shape[-1]**0.5,dim=-1)

        context_vec = attn_weights @ value
        return context_vec
    

batch_data = torch.stack((input1,input1),dim=0)
ca = CasualAttention(8,4,10,2,0.1)
logits=  ca(batch_data)
print(logits)