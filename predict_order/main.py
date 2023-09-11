
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn.functional as F
import itertools
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import ray

from Logger import logger
from Trainer import Trainer


#numpy random generator
rng = np.random.default_rng()
datetime_now = datetime.datetime.now()
datetime_now = "{0:%Y-%m-%d %H-%M-%S}".format(datetime_now)
ray.init()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
# ハイパーパラメータなどの設定値
hp_table={}
num_epochs = 5000 # 学習を繰り返す回数
num_batch = 24*20000   # 一度に処理する画像の枚数,class数を均一にするため24の倍数
learning_rate = 0.0005   # 学習率
weight_decay=0.1
max_data=10**6  #使うデータ数の上限
gradation_alpha=0.1 #順位1変化を考慮する割合
train_size=0.7
noise_delta=1
change_label_rate=1

is_log=True
save_model=True
model=None
alpha_prior=None


memo=''


hp_table['device']=device
hp_table['num_epochs']=num_epochs
hp_table['num_batch']=num_batch
hp_table['learning_rate']=learning_rate
hp_table['weight_decay']=weight_decay
hp_table['max_data']=max_data
hp_table['gradation_alpha']=gradation_alpha
hp_table['train_size']=train_size
hp_table['is_log']=is_log
hp_table['model']=model
hp_table['alpha_prior']=alpha_prior
hp_table['noise_delta']=noise_delta
hp_table['change_label_rate']=change_label_rate
hp_table['memo']=memo
class Args:
    def __init__(self,dic):
        self.is_log=dic['is_log']
        self.train_size=dic['train_size']
        self.gradation_alpha=dic['gradation_alpha']
        self.num_batch=dic['num_batch']
        self.learning_rate=dic['learning_rate']
        self.weight_decay=dic['weight_decay']
        self.num_epochs=dic['num_epochs']
        self.device=dic['device']
        self.max_data=dic['max_data']
        self.model=dic['model']
        self.alpha_prior=dic['alpha_prior']
        self.noise_delta=dic['noise_delta']
        self.change_label_rate=dic['change_label_rate']

        self.memo=dic['memo']

args=Args(hp_table)

hp_table = [f'| {key} | {value} |' for key, value in hp_table.items()]


writer=None
if is_log:
    writer = logger.remote(f'result_log/{datetime_now}')
    writer.add_text.remote(
    "Hyperparameters",
    "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)


ns=[3,4]#numOfPeople
rules=['ton','nan']

result=[]

for n in ns:
    for ri,r in enumerate(rules):
        rule=f'{n}{r}'
        npy=np.load(f'../npy/{rule}.npy')

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
            kt=(k//3)*4+(k%3) if n==3 else k
            knpy=npy[np.where(npy[:,0]==kt)]
            knpy=knpy[:,1:].astype(np.float32)
            knpy=rng.choice(knpy,size=min(max_data,knpy.shape[0]),replace=False)
            print(f'{rule}{k}局:{knpy.shape[0]}局')


            
            if save_model:
                save_dir=f'models/{datetime_now}/'
                os.makedirs(save_dir, exist_ok=True)
                save_model_dir_path=f'models/{datetime_now}/{n}{r}_{k}model_weights.pth'
            else:
                save_model_dir_path=None

            trainer=Trainer.remote(n,r,k,args,knpy,logger=writer,save_model_dir_path=save_model_dir_path)
            result.append(trainer.training.remote())



get_result=ray.get(result)#GPUはメモリ的にきつい
if is_log:writer.close.remote()
print('ALL Complete!')