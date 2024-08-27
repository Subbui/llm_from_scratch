import os
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

dataset = load_dataset("ccdv/pubmed-summarization","document",split="train[:10]")
dataset = " ".join(dataset['abstract'])
# tokenizer = tiktoken.get_encoding('cl100k_base')

class PrepareData(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        super().__init__()
        self.inputids = []
        self.targetids = []

        data_ids = tokenizer.encode(text)

        for i in range(0,len(data_ids)-max_length,stride):
            inputid = data_ids[i:i+max_length]
            targetid = data_ids[i+1:i+max_length+1]

            self.inputids.append(torch.tensor(inputid))
            self.targetids.append(torch.tensor(targetid))

    def __getitem__(self, index):
        return self.inputids[index], self.targetids[index]
    
    def __len__(self):
        return len(self.inputids)
    
def dataload(text,num_batches,context_len,stride,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    data_set = PrepareData(text,tokenizer,max_length=context_len,stride=stride)
    load_data = DataLoader(dataset=data_set,
                           batch_size=num_batches,
                           num_workers=num_workers,
                           drop_last=drop_last)
    return load_data
    
data = dataload(dataset,num_batches=4,context_len=8,stride=2)
data = iter(data)
input1,target1 = next(data)


