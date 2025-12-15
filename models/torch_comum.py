

import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
#https://github.com/srishti-git1110/Lets-go-deep-with-PyTorch/blob/main/Dataset%20and%20DataPipes%20blog/Dataset_and_DataPipe_colab.ipynb
'''class KaggleImageCaptioningDataset(Dataset):
    def __init__(self, train_captions, root_dir, transform=None, bert_model='distilbert-base-uncased', max_len=512):
        self.df = pd.read_csv(train_captions, header=None, sep='|')
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_len = max_len

        self.images = self.df.iloc[:, 0]
        self.captions = self.df.iloc[:, 2]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.images[idx]
        path_to_image = os.path.join(self.root_dir, image_id)
        image = Image.open(path_to_image).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        tokenized_caption = self.tokenizer(caption,
                                           padding='max_length',  # Pad to max_length
                                           truncation=True,  # Truncate to max_length
                                           max_length=self.max_len,
                                           return_tensors='pt')['input_ids']

        return image, tokenized_caption'''
# baseado em https://discuss.pytorch.org/t/load-dataframe-in-torch/47436/2
class csvDataSet(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file, header='infer')  # iterator=True
        self.inputs = self.frame.loc[:, self.frame.columns != 'Salary']
        #df.drop(['c', 'd'], axis=1)  df[['c', 'd']].copy()
        self.outputs =  self.frame[['Salary']].copy()
        #self.frame = pd.read_csv(csv_file, header='infer')# iterator=True
        print(self.frame)

    def __len__(self):
        # print len(self.landmarks_frame)
        # return len(self.landmarks_frame)
        return len(self.frame)

    def __getitem__(self, idx):
        print("i-:",idx)
        x = torch.tensor(self.inputs[idx], dtype=torch.float)
        #x = torch.tensor(self.inputs[idx], dtype=torch.int)
        y = torch.tensor(np.argmax(self.outputs[idx]))
        #landmarks = self.frame.get_chunk(128).as_matrix().astype('float')
        # landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')

        #
        return x,y


filename = 'data/Position_Salaries.csv'
dataset = csvDataSet(filename)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
for data in train_loader:
    print(data)