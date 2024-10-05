from gpt import generate_text, GPT
import tiktoken
import torch
from rough import calc_loss_batch, calc_loss_loader
from gpt import GPT
from dataloader import DataLoader_v1

config_gpt = {
    'vocab' : 50257,
    'context_length':256,
    'emb_dim':768,
    'num_heads':12,
    'num_layers':12,
    'drop_rate':0.1,
    'qkv_bias':False
}
with open(r"C:\Subbu\llms_from_scratch\data\verdict.txt",'r',encoding='utf-8') as f:
    data = f.read()
tokenizer = tiktoken.get_encoding('gpt2')

train_data = data[:int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]

train_loader = DataLoader_v1(train_data,context_length=config_gpt['context_length'],
                                stride=config_gpt['context_length'],
                                num_batches=2,
                                shuffle=True,
                                drop_last=True,
                                num_workers=0)


val_loader = DataLoader_v1(test_data,context_length=config_gpt['context_length'],
                                stride=config_gpt['context_length'],
                                num_batches=2,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)

tokenizer = tiktoken.get_encoding('gpt2')

def text_to_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={"<|endoftext|"})
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded

def ids_to_text(ids,tokenizer):
    ids = ids.squeeze(0).tolist()
    return tokenizer.decode(ids)

model = GPT(config_gpt)
start_context = "Every effort moves you"
text = generate_text(model=model,
                     idx=text_to_ids(start_context,tokenizer),
                     max_new_tokens=10,
                     context_length=config_gpt['context_length'])


def train_gpt(model,train_loader,val_loader,optimizer,device,num_epoch,eval_freq,eval_iter,start_context,tokenizer):
    total_train_loss,total_val_loss,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1
    for epoch in range(num_epoch):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step +=1

            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                total_train_loss.append(train_loss)
                total_val_loss.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:02d}): "f"Train loss {train_loss:.3f}, Validation loss {val_loss:.3f}")

        generate_and_print(model,tokenizer,device,start_context)
    return total_train_loss, total_val_loss, track_tokens_seen

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss



def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
        model=model, idx=encoded,
        max_new_tokens=15, context_length=context_size
        )
        decoded_text = ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)
model = GPT(config_gpt)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A
num_epochs = 10
train_losses, val_losses, tokens_seen = train_gpt(
    model, train_loader, val_loader, optimizer, device,
    num_epoch=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Life is all about", tokenizer=tokenizer)

# total_parameters = sum(x.numel() for x in model.parameters())
# final_params = total_parameters - (sum(p.numel() for p in model.out_layer.parameters()))
# print(total_parameters,final_params)

# torch.save(model.state_dict(),'model.pth')

torch.save({
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict()
},'C:\Subbu\llms_from_scratch\model\model_optimizer.pth')