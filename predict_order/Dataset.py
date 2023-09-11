import numpy as np
import math
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,data_x,data_y,batch_size,numOfPeople,dimy,dtf,label_gradation=0.1,noize_delta=1,change_label_rate=0.05):
        super().__init__()
        self.label_index=[torch.where(data_y==i)[0]for i in range(dimy)]
        self.label_length=[len(self.label_index[i]) for i in range(dimy)]
        self.length=dimy*max(self.label_length)
        for i in range(dimy):
            r=torch.randperm(self.label_length[i])
            self.label_index[i]=self.label_index[i][r]

        self.data_length=len(data_y)
        self.data_x=data_x
        self.data_y=F.one_hot(data_y,num_classes=dimy)
        self.data_y=self.data_y.to(torch.float32)
        self.batch_size=batch_size
        self.numOfPeople=numOfPeople
        self.label_gradation=label_gradation
        self.dimy=dimy
        
        mt=350 if self.numOfPeople==3 else 250
        self.noise_delta=noize_delta/(mt+2)/self.numOfPeople#第１項 = 1　だと整数範囲では順位に影響なし
        self.noise_delta = self.noise_delta

        self.change_label_rate=change_label_rate
        self.target_label=0
        self.dtf=dtf
    
    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        label=index%self.dimy
        if self.label_length[label]==0:
            random_index=np.random.random_integers(self.data_length-1)
            x=self.data_x[random_index]
            y=self.data_y[random_index]
            random_label=torch.argmax(y).item()
            x=self.dtf.change_label(x,random_label,label)
            y=torch.zeros(self.dimy)
            y[label]=1

            
        else:
            index_l=(index//self.dimy)%self.label_length[label]
        
            x=self.data_x[self.label_index[label][index_l]]
            y=self.data_y[self.label_index[label][index_l]]
            
            

        if np.random.rand()<self.change_label_rate:
            x_copy=x.detach().clone()
            x=self.dtf.change_label(x,label,self.target_label)
            y=torch.zeros(self.dimy)
            y[self.target_label]=1
            
            self.target_label = (self.target_label +1 )%self.dimy
            
            
        noise=torch.rand(x.shape)
        noise=(noise-0.5)*self.noise_delta
        x=x+noise


        if index_l+1==self.label_length[label]:
            r=torch.randperm(self.label_length[label])
            self.label_index[label]=self.label_index[label][r]
        return x,y

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self) -> int:
        return self.length