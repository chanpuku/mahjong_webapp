import numpy as np
#import os
import math
import torch
import ray
from sklearn.model_selection import train_test_split
import time

from util import MyLoss,EarlyStopping,elapsed_time_str
from Model import MyNet
from Dataset import MyDataset
from DataTransForm import DataTransform

@ray.remote#(num_gpus=0.1)
class Trainer:
    """
    args:
    is_log
    train_size
    gradation_alpha(=0ならなし)
    num_batch
    learning_rate
    num_epochs
    device

    npy:np.float32
        shape=[m,n+n]#持ち点+順位
    """
    def __init__(self,n,rule,kyoku,args,npy,logger=None,save_model_dir_path=None):
        self.args=args
        self.n=n
        self.rule=rule
        self.kyoku=kyoku
        self.logger=logger
        self.npy=npy
        self.dtf=DataTransform(n,args.gradation_alpha)
        self.dimy=math.factorial(n)

        input_size=n*(n+1)//2
        if args.model==None:
            self.model=MyNet(input_size,self.dimy).to(args.device)
        self.is_log=False if logger==None else True
        self.save_model_dir_path=save_model_dir_path


    def training(self):
        train, test = train_test_split(self.npy,train_size=self.args.train_size)
        train_x,train_y=train[:,:self.n],train[:,self.n:]
        test_x,test_y=test[:,:self.n],test[:,self.n:]
        train_x,train_y=self.dtf.make_data(train_x,train_y,self.kyoku,onehot=False)
        test_x,test_y=self.dtf.make_data(test_x,test_y,self.kyoku,onehot=True)

        mydataset=MyDataset(train_x, train_y,self.args.num_batch,self.n,self.dimy,self.dtf,
                            label_gradation=self.args.gradation_alpha,noize_delta=self.args.noise_delta,change_label_rate=self.args.change_label_rate)

        print(f'{self.n}{self.rule}{self.kyoku}局:{mydataset.label_length}')
        train_dataloader = torch.utils.data.DataLoader(
            mydataset,
            batch_size = self.args.num_batch,
            shuffle = False)
        test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_x, test_y), 
            batch_size = self.args.num_batch,
            shuffle = True)
        
        criterion = MyLoss(self.dimy)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.learning_rate,weight_decay=self.args.weight_decay) 
        gradation_matrix=self.dtf.gradation_matrix.to(self.args.device)
        
        self.model.train()
        training_start_time=time.time()

        print_log_step=min(self.args.num_epochs//5,100)
        test_log_step=10
        earlystopping = EarlyStopping(patience=30, verbose=False,path=self.save_model_dir_path) #test_timingのみ

        for epoch in range(self.args.num_epochs): 
            loss_sum = 0
            for x, y in train_dataloader:
                x=x.to(self.args.device)
                y=y.to(self.args.device)
                # optimizerを初期化
                optimizer.zero_grad()
                outputs = self.model(x)
                targets = torch.mm(y,gradation_matrix)
                targets /=torch.sum(targets,dim=1,keepdim=True)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss
            
            

            train_loss=loss_sum.item() / len(train_dataloader)
            if self.is_log:self.logger.add_scalar.remote(f'train_loss/{self.n}{self.rule}{self.kyoku}',train_loss,epoch)

            if epoch%print_log_step==0:
                # 学習状況の表示
                sum_time=time.time()-training_start_time
                pred_sum_time=sum_time/(epoch+1)*(self.args.num_epochs)
                print(f"[{self.n}{self.rule}{self.kyoku}]Epoch: {epoch+1}/{self.args.num_epochs},Time:{elapsed_time_str(sum_time)}/{elapsed_time_str(pred_sum_time)}, Loss: {train_loss}")
                

            if epoch%test_log_step==0:
                self.model.eval()
                test_loss_sum = 0
                test_gradation_loss_sum=0
                test_diri_loss_sum=0
                test_diri_gradation_loss_sum=0

                with torch.no_grad():
                    for x, y in test_dataloader:
                        x = x.to(self.args.device)
                        y = y.to(self.args.device)

                        outputs = self.model(x)
                        outputs_p=outputs/torch.sum(outputs,dim=1,keepdim=True)
                        test_loss_sum += -1*torch.mean(torch.sum(torch.log(outputs_p)*y,dim=1))

                        y_eps=y+1e-24
                        y_eps/=torch.sum(y_eps,dim=1,keepdim=True)
                        test_diri_loss_sum += criterion(outputs, y_eps)

                        targets =torch.mm(y,gradation_matrix)
                        targets /=torch.sum(targets,dim=1,keepdim=True)
                        test_gradation_loss_sum  += -1*torch.mean(torch.sum(torch.log(outputs_p)*targets,dim=1))
                        test_diri_gradation_loss_sum  += criterion(outputs, targets)
                self.model.train()
                test_loss=test_loss_sum.item() / len(test_dataloader)
                test_gradation_loss=test_gradation_loss_sum.item() / len(test_dataloader)
                test_diri_loss=test_diri_loss_sum.item() / len(test_dataloader)
                test_diri_gradation_loss=test_diri_gradation_loss_sum.item() / len(test_dataloader)
               
                if self.is_log:
                    self.logger.add_scalar.remote(f'test_loss/{self.n}{self.rule}{self.kyoku}',test_loss,epoch)
                    self.logger.add_scalar.remote(f'test_gradation_loss/{self.n}{self.rule}{self.kyoku}',test_gradation_loss,epoch)
                    self.logger.add_scalar.remote(f'test_diri_loss/{self.n}{self.rule}{self.kyoku}',test_diri_loss,epoch)
                    self.logger.add_scalar.remote(f'test_diri_gradation_loss/{self.n}{self.rule}{self.kyoku}',test_diri_gradation_loss,epoch)
                
                earlystopping(test_loss, self.model) #callメソッド呼び出し
                if earlystopping.early_stop: 
                    print("Early Stopping!")
                    break
        print(f"[{self.n}{self.rule}{self.kyoku}]Complete")
        return 0
