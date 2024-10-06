from gpt import GPT
# from gpt_train import config_gpt
import torch
config_gpt = {
    'vocab' : 50257,
    'context_length':256,
    'emb_dim':768,
    'num_heads':12,
    'num_layers':12,
    'drop_rate':0.1,
    'qkv_bias':False
}
# model = GPT(config_gpt)
# model.load_state_dict(torch.load('C:\Subbu\llms_from_scratch\model\model.pth'))
# model.eval()

#in case we want to train the model later
checkpoint = torch.load("C:\Subbu\llm_from_scratch\model\model_optimizer.pth")
model = GPT(config_gpt)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer =  torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# model.train()
print(sum(p.numel() for p in model.parameters()))