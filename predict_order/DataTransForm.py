import numpy as np
import math
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F


class DataTransform:
    def __init__(self,numOfPeople,gradation_alpha):
        self.numOfPeople=numOfPeople
        self.dimy=math.factorial(numOfPeople)
        odic={}
        self.order_list=list(itertools.permutations(range(numOfPeople)))
        for i in range(len(self.order_list)):
            odic[tuple(self.order_list[i])]=i
        self.odic=odic

        self.change_label_dic={}#交換先のindexを返す,新しいベクトルはx[li]とすればよい
        for i in range(len(self.order_list)):
            oi=self.order_list[i]
            for j in range(len(self.order_list)):
                ll=[0]*self.numOfPeople
                oj=self.order_list[j]
                for k in range(self.numOfPeople):
                    kc=0
                    for ki in range(self.numOfPeople):
                        if oi[ki]==oj[k]:
                            kc=ki
                            break
                    ll[k]=kc

                self.change_label_dic[(i,j)]=ll

        self.gradation_matrix=self.make_gradation_matrix(gradation_alpha)
    def apply_dic(self,array):
        return self.odic[tuple(array)]
    
    def make_data(self,x,y,kyoku,onehot=True):
        xx=self.make_x(x,kyoku)
        yy=self.make_y(y,onehot=onehot)
        return xx,yy
    def make_x(self,array,kyoku,dif=False):#(n,4or3)
        mt=350 if self.numOfPeople==3 else 250
        a=array/(mt+1)
        dt=1/(mt+1)/self.numOfPeople
        #点数の調整
        for i in range(self.numOfPeople-1):
            a[:,(i-kyoku)%self.numOfPeople]+=dt*(self.numOfPeople-1-i)

        dda=self.numOfPeople*(self.numOfPeople-1)//2
        da=np.zeros((a.shape[0],dda))

        c=0
        for i in range(1,self.numOfPeople):
            for j in range(i):
                da[:,c]=a[:,i]-a[:,j]
                c+=1
        a=np.concatenate([a,da],axis=1)
        a= torch.tensor(a,dtype=torch.float32)
        return a
    def make_y(self,array,onehot=False):
        data_y=np.apply_along_axis(self.apply_dic, 1, array)
        data_y=torch.tensor(data_y,dtype=torch.int64)
        if onehot:
            data_y=F.one_hot(data_y,num_classes=self.dimy)
            data_y=data_y.to(torch.float32)
        return data_y
    def change_label(self,x,label,target_label):
        tv=self.change_label_dic[(label,target_label)]
        with torch.no_grad():
            xx=x[:self.numOfPeople][tv]
            x[:self.numOfPeople]=xx
            c=0
            for i in range(1,self.numOfPeople):
                for j in range(i):
                    x[self.numOfPeople+c]=x[i]-x[j]
                    c+=1
        return x
    def make_gradation_matrix(self,gradation_alpha):
        mat=[[np.inf]*self.dimy for i in range(self.dimy)]
        for i in range(self.dimy):
            mat[i][i]=0
        for k,v in self.odic.items():
            for i in range(self.numOfPeople-1):
                lk=list(k)
                temp=lk[i+1]
                lk[i+1]=lk[i]
                lk[i]=temp
                idx=self.odic[tuple(lk)]
                mat[v][idx]=1
        for k in range(self.dimy):#ワーシャルフロイド
            for i in range(self.dimy):
                for j in range(self.dimy):
                    mat[i][j]=min(mat[i][j],mat[i][k]+mat[k][j])
        mat=torch.tensor(mat, dtype=torch.float32)
        mat=gradation_alpha**mat
        return mat/torch.sum(mat,dim=1)
