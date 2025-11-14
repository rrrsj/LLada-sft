import pytorch_lightning as L
from torch.utils.data import DataLoader
from dataloader.mydataset import CustomDataset
class mydataloader(L.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.args=args
    
    def setup(self,stage):
        if self.trainer is not None:
            rank_id = self.trainer.global_rank
            world_size = self.trainer.world_size
            print(f"[DataModule] Rank: {rank_id}, World Size: {world_size}")
        else:
            global_rank = 0
            world_size = 1
        self.train_dataset=CustomDataset(self.args,rank_id,world_size,'train')
        self.vali_dataset=CustomDataset(self.args,rank_id,world_size,'vali')
        self.test_dataset=CustomDataset(self.args,rank_id,world_size,'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.args['batch_size'],shuffle=False)

    # def vali_dataloader(self):
    #     return DataLoader(self.vali_dataset,batch_size=self.args['batch_size'],shuffle=False)

if __name__=='__main__':
    dataset=CustomDataset(self.args,rank_id,world_size,'train')