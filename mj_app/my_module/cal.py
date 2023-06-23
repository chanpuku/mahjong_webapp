import numpy as np
import torch
import itertools
import math
import itertools
import torch.nn.functional as F
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
    def make_x(self,array,tensor=True):#(n,4or3)
        a=np.array([array])/10000
        if tensor:
            a = torch.tensor(a, dtype=torch.float32)
        return a

class Cal:
    def __init__(self):
        self.my_models={}
        self.tfs=[TransForm(3),TransForm(4)]

    def predict(self,n,r,k,array):
        tf=TransForm(n)
        dimy=math.factorial(n)
        model=Net(n,dimy)
        model.load_state_dict(torch.load(f'./mj_app/my_module/models/{n}{r}_{k}model_weights.pth',map_location=torch.device('cpu')))
        data=tf.make_x(array)
        model.eval()
        with torch.no_grad():
            pred=model(data)
            pred=F.softmax(pred, dim=1)*100

        result=np.zeros((n,n))
        for t in itertools.permutations(range(n)):
            for j in range(n):
                result[j,t[j]]+=pred[0,tf.odic[t]]
        return result






