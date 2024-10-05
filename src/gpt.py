import torch
import torch.nn as nn
import tiktoken
from dataloader import DataLoader_v1
from multiheadattention import MultiHeadAttention

gpt_124M_config = {

    'vocab': 50257,
    'context_length':1024,
    "emb_dim":768,
    "num_heads":12,
    "num_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

class GPT(nn.Module):
    def __init__(self,config): 
        super().__init__()
        self.tok_emb = nn.Embedding(config['vocab'],config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'],config['emb_dim'])
        self.final_norm = LayerNorm(config['emb_dim'])
        self.trf_blk = nn.Sequential(*[TransformerBlock(config) for _ in range(config['num_layers'])])
        self.out_layer = nn.Linear(config['emb_dim'],config['vocab'],bias=False)
        self.drop_emb = nn.Dropout(config['drop_rate'])

    def forward(self,x):
        batch,seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len,device=x.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blk(x)
        x = self.final_norm(x)
        logits = self.out_layer(x)
        return logits




class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
         return 0.5 * x * (1 + torch.tanh(
             torch.sqrt(torch.tensor(2.0 / torch.pi)) *
             (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config['emb_dim'],4*config['emb_dim']),
                                    GELU(),
                                    nn.Linear(4*config['emb_dim'],config['emb_dim']))
    def forward(self,x):
         return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True, unbiased=False)
        norm = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm+self.shift
    
class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = config['emb_dim'],
            d_out = config['emb_dim'],
            num_heads= config['num_heads'],
            context_length=config['context_length'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias'])
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_sc = nn.Dropout(config['drop_rate'])
        self.ff = FeedForward(config)
    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_sc(x)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_sc(x)
        x = x + shortcut
        return x



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_text(model,idx,max_new_tokens,context_length):
    for _ in range(max_new_tokens):
        idx_contd = idx[:,-context_length:]
        with torch.no_grad():
            logits= model(idx_contd)

        logits = logits[:,-1,:]
        proba = torch.softmax(logits,dim=-1)
        id_next = torch.argmax(proba,dim=-1,keepdim=True)
        idx = torch.cat((idx,id_next),dim=1)

    return idx 

# start_context = "Hello, I am"
# tok = tiktoken.get_encoding('gpt2')
# ids = tok.encode(start_context)
# # print(ids)
# ids = torch.tensor(ids).unsqueeze(0)
# # print(ids.shape)

# model = GPT(gpt_124M_config)
# model.eval()
# text = generate_text(model,ids,6,1024)
# print(tok.decode(text.squeeze(0).tolist()))


# with open(r"C:\Subbu\llms_from_scratch\data\verdict.txt",'r') as f:
#     data = f.read()

# data = data[:100]

# dataload = DataLoader_v1(data,shuffle=False)
# data_iter = iter(dataload)
# inputids,targetids = next(data_iter)

# gpt = GPT(gpt_124M_config)
# gpt_output = gpt(inputids)
# print(gpt_output)
     
# input_batch = torch.tensor([[ 6109, 3626, 6100, 345], # token IDs of text 1
# [ 6109, 1110, 6622, 257]])
# torch.manual_seed(123)
# model = GPT(gpt_124M_config)
# out = model(input_batch)
# print("Input batch:\n", input_batch)
# print("\nOutput shape:", out.shape)
# print(out)

# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

# torch.manual_seed(123)
# model = GPT(gpt_124M_config)
# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

# total_params = sum(p.numel() for p in model.parameters())
# print(total_params)

