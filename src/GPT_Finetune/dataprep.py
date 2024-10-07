import pandas as pd
import tiktoken
from torch.utils.data import Dataset,DataLoader
import torch

data = pd.read_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\SMSSpamCollection.tsv',delimiter='\t',names=['Labels','Text'])
# print(data['Labels'].value_counts())

#Make the data balanced
spam = data[data['Labels']=='spam'].shape[0]
ham = data[data['Labels']=='ham'].sample(spam,random_state=123)
data_final = pd.concat((ham,data[data['Labels']=='spam']))
data_final['Labels'] = data_final['Labels'].map({'ham':0,'spam':1})
# print(data_final.shape)
print(data_final['Labels'].value_counts())

def data_split(data,train_frac,val_frac):
    train_end = int(len(data)*train_frac)
    val_end = train_end + int(len(data)*val_frac)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

train_data,val_data,test_data= data_split(data_final,0.7,0.1)
# print(train_data.shape, val_data.shape, test_data.shape)

# train_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\trian.csv')
# val_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\val.csv')
# test_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\test.csv')


#prepare data loaders

tokenizer = tiktoken.get_encoding('gpt2')
# print(tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"}))

class SpamDataLoader(Dataset):
    def __init__(self,csv_file,tokenizer,max_length=None,pad_token=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_text = [tokenizer.encode(text) for text in self.data['Text']]

        if max_length is None:
            self.max_length = self._get_max_length()
        else:
            self.max_length = max_length
        
        self.encoded_text = [encoded_text[:self.max_length] for encoded_text in self.encoded_text]
        self.encoded_text = [encoded_text + (self.max_length - len(encoded_text))*[pad_token] for encoded_text in self.encoded_text]
        
    def __getitem__(self,idx):
        text = self.encoded_text[idx]
        labels = self.data.iloc[idx]['Labels']
        return (torch.tensor(text,dtype=torch.long), torch.tensor(labels,dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def _get_max_length(self):
        max_length = 0
        for encoded_text in self.encoded_text:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length
    
train_data = SpamDataLoader(r"C:\Subbu\llm_from_scratch\data\sms_spam_collection\trian.csv",tokenizer)
val_data = SpamDataLoader(r"C:\Subbu\llm_from_scratch\data\sms_spam_collection\val.csv",tokenizer,max_length=train_data.max_length)
test_data = SpamDataLoader(r"C:\Subbu\llm_from_scratch\data\sms_spam_collection\test.csv",tokenizer,max_length=train_data.max_length)

num_workers=0
batch_size=8
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=True)
val_loader = DataLoader(dataset=val_data,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False)
test_loader = DataLoader(dataset=test_data,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False)

print(len(train_loader),len(val_loader),len(test_loader))
