import torch
import torch.nn as nn
import tiktoken
from attention import MultiHeadAttention

config_124M = {
    'vocab' : 100025,
    'context_length':1024,
    'emb_dim':768,
    'num_heads':12,
    'num_layers':12,
    'drop_rate':0.1,
    'qkv_bias':False
}

class GPTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_emb = nn.Embedding(config['vocab'],config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'],config['emb_dim'])
        self.layer_norm = LayerNorm(config['emb_dim'])
        self.trfblk = nn.Sequential(*[TransformerBlock(config) for _ in range(config['num_layers'])])
        self.out_head = nn.Linear(config['emb_dim'],config['vocab'])
        self.dropout = nn.Dropout(config['drop_rate'])

    def forward(self,x):
        batch_size,seq_length = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_length,device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)

        x = self.trfblk(x)
        x = self.layer_norm(x)
        x = self.out_head(x)

        return x
    
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True)
        var = torch.var(x,dim=-1,keepdim=True)
        norm = x-mean/(torch.sqrt(var+self.eps))
        return self.scale*norm+self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config['emb_dim'],4*config['emb_dim'],GELU()),
                                    nn.Linear(4*config['emb_dim'],config['emb_dim']))
    def forward(self,x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.norm1 = LayerNorm(config['emb_dim'])
        self.attn_blk = MultiHeadAttention(
            d_in = config['emb_dim'],
            d_out = config['emb_dim'],
            context_length=config['context_length'],
            num_heads = config['num_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.norm2 = LayerNorm(config['emb_dim'])
        self.out_layer = FeedForward(config)
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn_blk(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x =self.out_layer(x)
        x = self.drop_shortcut(x)
        x = x+shortcut
        return x
    



    def forward(self,x):
        return x
    
tokenizer = tiktoken.get_encoding("cl100k_base")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch,dim=0)

# gptmodel = GPTModel(config_124M)
# output = gptmodel(batch)
# print(output)

# total_parameters = sum(p.numel() for p in gptmodel.parameters())
# print(total_parameters)
# print(gptmodel.token_emb.weight.shape)   
# print(gptmodel.out_head.weight.shape)

# print(batch.shape)
# print(batch.unsqueeze(1).shape)