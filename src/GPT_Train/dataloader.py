import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from importlib.metadata import version

tokenizer = tiktoken.get_encoding('gpt2')

class DatasetPrep(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        super().__init__()
        self.inputids = []
        self.targetids = []
        ids = tokenizer.encode(text)

        for i in range(0,len(ids)-max_length,stride):
            input_ids = ids[i:i+max_length]
            target_ids = ids[i+1:i+max_length+1]
            
            self.inputids.append(torch.tensor(input_ids))
            self.targetids.append(torch.tensor(target_ids))

    def __getitem__(self,idx):
        return self.inputids[idx], self.targetids[idx]
    
    def __len__(self):
        return len(self.inputids)
    
def DataLoader_v1(text,context_length=6,stride=3,num_batches=3,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = DatasetPrep(text,tokenizer,context_length,stride)
    dataload = DataLoader(
        dataset=dataset,
        batch_size=num_batches,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataload
    





