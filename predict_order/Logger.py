import ray
from torch.utils.tensorboard import SummaryWriter

@ray.remote
class logger:
    def __init__(self,dir):
        self.writer = SummaryWriter(dir)
    def add_text(self,t1,t2):
        self.writer.add_text(t1,t2)
    
    def add_scalar(self,dir,y,x):
        self.writer.add_scalar(dir,y,x)

    def close(self):
        self.writer.close()