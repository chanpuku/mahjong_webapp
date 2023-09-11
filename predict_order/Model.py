import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNet, self).__init__()

        hidden_dim0=output_size*20
        hidden_dim1=output_size*10
        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, hidden_dim0)
        self.fc2 = nn.Linear(hidden_dim0, hidden_dim0)
        self.fc3 = nn.Linear(hidden_dim0, hidden_dim0)
        self.fc4 = nn.Linear(hidden_dim0,hidden_dim0)
        self.fc5 = nn.Linear(hidden_dim0,hidden_dim1)
        self.fc6 = nn.Linear(hidden_dim1,hidden_dim1)
        self.fc7 = nn.Linear(hidden_dim1,hidden_dim1)
        self.fc8 = nn.Linear(hidden_dim1,hidden_dim1)
        self.fc9 = nn.Linear(hidden_dim1,hidden_dim1)
        self.fc10 = nn.Linear(hidden_dim1,output_size)

        self.dropout= nn.Dropout(0.2)
        self.bn4 =nn.BatchNorm1d(hidden_dim0)
        self.bn6 =nn.BatchNorm1d(hidden_dim1)
        self.bn8 =nn.BatchNorm1d(hidden_dim1)
        self.gelu=nn.GELU()
        self.elu=nn.ELU()

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)

        shortcut = x
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.gelu(x)
        x = x + shortcut
        
        x = self.fc5(x)
        x = self.gelu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.gelu(x)

        shortcut = x
        x = self.fc7(x)
        x = self.gelu(x)
        x = self.fc8(x)
        x = self.bn8(x)
        x = self.gelu(x)
        x = x + shortcut

        x = self.fc9(x)
        x = self.gelu(x)
        x = self.fc10(x)
        x = self.elu(x)+1
        return x