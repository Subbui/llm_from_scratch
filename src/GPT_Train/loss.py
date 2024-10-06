import torch
import tiktoken
from src.GPT_Train.gpt import GPT
# from gpt_train import config_gpt
from src.GPT_Train.dataloader import DataLoader_v1





def calc_loss_batch(inputids,targetids,model,device):
    inputids,targetids = inputids.to(device), targetids.to(device)
    logits = model(inputids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),targetids.flatten())
    return loss

def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader),num_batches)
    
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i < num_batches:
            calc_loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += calc_loss.item()
        else:
            break
    return total_loss/num_batches

# model = GPT(config_gpt)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader,model,device)
#     val_loss = calc_loss_loader(val_loader,model,device)

# print(f'train_loss: {train_loss}')
# print(f'val loss: {val_loss}')




# inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
# [40, 1107, 588]]) # "I really like"]
# targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
# [107, 588, 11311]]) # " really like chocolate"]

# model = GPT(config_gpt)
# torch.manual_seed(123)
# with torch.no_grad():
#     logits = model(inputs)
# proba = torch.softmax(logits,dim=-1)

# token_ids = torch.argmax(proba,dim=-1,keepdim=True)
# # print(token_ids[0])
# idx=0
# print(proba.shape)
# print(proba[idx,[0,1,2],targets[idx]])