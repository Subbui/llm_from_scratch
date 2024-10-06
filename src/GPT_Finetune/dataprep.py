import pandas as pd
import tiktoken

data = pd.read_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\SMSSpamCollection.tsv',delimiter='\t',names=['Labels','Text'])
# print(data['Labels'].value_counts())

#Make the data balanced
spam = data[data['Labels']=='spam'].shape[0]
ham = data[data['Labels']=='ham'].sample(spam,random_state=123)
data_final = pd.concat((ham,data[data['Labels']=='spam']))
# print(data_final.shape)
# print(data_final['Labels'].value_counts())

def data_split(data,train_frac,val_frac):
    train_end = int(len(data)*train_frac)
    val_end = train_end + int(len(data)*val_frac)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

train_data,val_data,test_data= data_split(data_final,0.7,0.1)
print(train_data.shape, val_data.shape, test_data.shape)

# train_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\trian.csv')
# val_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\val.csv')
# test_data.to_csv(r'C:\Subbu\llm_from_scratch\data\sms_spam_collection\test.csv')




# tokenizer = tiktoken.get_encoding('gpt2',allowed_special={'<|endoftext|>'})
# print(tokenizer.encode('<|endoftext|>'))
