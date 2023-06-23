import numpy as np
#import os
import math
#from autograd import grad
import torch
#from scipy.stats import entropy
import itertools
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, output_size)

        self.dropout= nn.Dropout(0.25)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransForm:
    def __init__(self,numOfPeople):
        self.numOfPeople=numOfPeople
        self.dimy=math.factorial(numOfPeople)
        odic={}
        l=list(itertools.permutations(range(numOfPeople)))
        for i in range(len(l)):
            odic[tuple(l[i])]=i
        self.odic=odic
    def apply_dic(self,array):
        return self.odic[tuple(array)]
    
    def make_data(self,x,y,kyoku,under_sampling=True,tensor=True):
        if under_sampling:
            xx=self.make_x(x,kyoku,tensor=False)
            yy=self.make_y(y,tensor=False)
            idxs=self.under_sampling(yy)
            xx=xx[idxs]
            yy=yy[idxs]
            if tensor:
                xx= torch.tensor(xx, dtype=torch.float32)
                yy=torch.tensor(yy, dtype=torch.int64)
        else:
            xx=self.make_x(x,kyoku,tensor=tensor)
            yy=self.make_y(y,tensor=tensor)
        return xx,yy
    def make_x(self,array,kyoku,dif=False,tensor=True):#(n,4or3)

        mt=350 if self.numOfPeople==3 else 250
        a=array/(mt+1)
        dt=1/(mt+1)/self.numOfPeople
        #点数の調整
        for i in range(self.numOfPeople-1):
            a[:,(i-kyoku)%self.numOfPeople]+=dt*(self.numOfPeople-1-i)

        if tensor:
            a = torch.tensor(a, dtype=torch.float32)
        return a
    def make_y(self,array,tensor=True):
        data_y=np.apply_along_axis(self.apply_dic, 1, array)
        #data_y=np.eye(self.dimy)[data_y] #onehot化
        if tensor:
            data_y= torch.tensor(data_y, dtype=torch.int64)
        return data_y
    
    def under_sampling(self,y):
        idxs=[0]*self.dimy
        for i in range(self.dimy):
            idxs[i]=np.where(y==i)[0]
        nums=[idxs[i].shape[0] for i in range(self.dimy)]
        min_num=min(nums)
        print(f'in under_sampling, num={min_num}')
        randidx=np.random.permutation(min_num)
        for i in range(self.dimy):
            idxs[i]=idxs[i][randidx]
        return np.concatenate(idxs)


def elapsed_time_str(seconds):
    """秒をhh:mm:ss形式の文字列で返す

    Parameters
    ----------
    seconds : float
        表示する秒数

    Returns
    -------
    str
        hh:mm:ss形式の文字列
    """
    seconds = int(seconds + 0.5)    # 秒数を四捨五入
    h = seconds // 3600             # 時の取得
    m = (seconds - h * 3600) // 60  # 分の取得
    s = seconds - h * 3600 - m * 60 # 秒の取得

    return f"{h:02}:{m:02}:{s:02}"  # hh:mm:ss形式の文字列で返す