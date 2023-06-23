
import numpy as np
#import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn.functional as F
import itertools
import torch.nn as nn
from my_module import TransForm
from my_module import Net
from my_module import elapsed_time_str as ets

from torch.utils.tensorboard import SummaryWriter

import time




#numpy random generator
rng = np.random.default_rng()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ハイパーパラメータなどの設定値
hp_table={}
num_epochs = 250  # 学習を繰り返す回数
num_batch = 5000   # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率
under_sampling=True
max_data=2*10**5  #使うデータ数の上限
hp_table['num_epochs']=num_epochs
hp_table['num_batch']=num_batch
hp_table['learning_rate']=learning_rate
hp_table['under_sampling']=under_sampling
hp_table['max_data']=max_data

hp_table = [f'| {key} | {value} |' for key, value in hp_table.items()]


log=True
test_log_step=10
train_print_step=10

ns=[3,4]#numOfPeople
rules=['ton','nan']



for n in ns:
    tf=TransForm(n)
    dimy=math.factorial(n)
    for ri,r in enumerate(rules):
        if n==3 and r=='ton':continue
        rule=f'{n}{r}'
        npy=np.load(f'npy/{rule}.npy')

        #k=0に限り0本場を削除
        npy=npy[np.where((npy[:,0]!=0)|(npy[:,1]!=0))]

        #npy=npy[np.where(npy[:,1]==0)]#本場
        #使うところだけ、局[0],持ち点[3:7],最終順位[14:18]
        idxs=[0]
        for i in range(2):
            for j in range(n):
                idxs.append(j+3+11*i)
        npy=npy[:,idxs].astype(np.int16)

        n_kyokus=n*(ri+2)
        for k in range(n_kyokus):
            if log:writer = SummaryWriter(f'result_log/{n}{r}{k}')
            if log:
                writer.add_text(
                "Hyperparameters",
                "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)

            kt=(k//3)*4+(k%3) if n==3 else k
            knpy=npy[np.where(npy[:,0]==kt)]

            


            print(f'{rule}{k}局:{knpy.shape[0]}局')
            print(f'k:{k},kt:{kt}')
            idxs=rng.permutation(knpy.shape[0])[:max_data]
            knpy=knpy[idxs,1:].astype(np.float32)
            print(f'knpy.shape:{knpy.shape}')
            train, test = train_test_split(knpy,train_size=0.7)
            train_x,train_y=train[:,:n],train[:,n:n*2]
            test_x,test_y=test[:,:n],test[:,n:n*2]
            train_x,train_y=tf.make_data(train_x,train_y,k,under_sampling=under_sampling)
            test_x,test_y=tf.make_data(test_x,test_y,k,under_sampling=False)

            train_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x, train_y),
                batch_size = num_batch,
                shuffle = True)
            test_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_x, test_y),  
                batch_size = num_batch,
                shuffle = True)


            model=Net(n,dimy).to(device)
            criterion = nn.CrossEntropyLoss() 
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

            model.train() 
            training_start_time=time.time()

            for epoch in range(num_epochs): 
                loss_sum = 0

                for x, y in train_dataloader:
                    x=x.to(device)
                    y=y.to(device)

                    # optimizerを初期化
                    optimizer.zero_grad()

                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss_sum += loss
                    loss.backward()
                    optimizer.step()

                train_loss=loss_sum.item() / len(train_dataloader)
                if log:writer.add_scalar('train_loss',train_loss,epoch)

                if epoch%train_print_step==train_print_step-1:
                    # 学習状況の表示
                    sum_time=time.time()-training_start_time
                    pred_sum_time=sum_time/(epoch+1)*(num_epochs)
                    print(f"Epoch: {epoch+1}/{num_epochs},Time:{ets(sum_time)}/{ets(pred_sum_time)}, Loss: {train_loss}")
                    

                if epoch%test_log_step==test_log_step-1:
                    model.eval() 
                    test_loss_sum = 0

                    with torch.no_grad():
                        for x, y in test_dataloader:
                            x = x.to(device)
                            y = y.to(device)

                            outputs = model(x)
                            test_loss_sum += criterion(outputs, y)
                    model.train()
                    test_loss=test_loss_sum.item() / len(test_dataloader)
                    print('---')
                    print(f"Test_Loss: {test_loss}")
                    if log:writer.add_scalar('test_loss',test_loss,epoch)
            # モデルの重みの保存
            torch.save(model.state_dict(), f'models/{rule}_{k}model_weights.pth')

            #定性的チェック
            yy=model(test_x[:100].to(device))
            print(f'yy.shape:{yy.size()}')
            yy_softmax=F.softmax(yy, dim=1)

            mt=351 if n==3 else 251
            for i in range(10):
                print('')
                print(test_x[i]*mt)
                rate=np.zeros((n,n))
                for pn in itertools.permutations(range(n)):
                    for j in range(n):
                        rate[j,pn[j]]+=yy_softmax[i,tf.odic[pn]]
                print(rate)
