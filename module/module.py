import torch
import torch.nn as nn
from utils.loss import cross_entropy_loss
import pytorch_lightning as L
from transformers import AutoTokenizer, AutoModel
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
import shutil
import os

class MyModule(L.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_temp=AutoModel.from_pretrained(self.args['default_checkpoint'],trust_remote_code=True)
        self.model=self.model_temp.model
        self.model.set_activation_checkpointing('whole_layer')
        self.write_config=0
        self.now_train_step=0
        self.now_vali_step=0
        self.now_test_step=0

    def training_step(self,batch,batch_idx):
        if self.write_config==0 and self.trainer.is_global_zero:
            self.write_config=1
            log_dir=f"./log/{str(self.args['save_time'])}"
            os.makedirs(log_dir,exist_ok=True)
            self.writer=SummaryWriter(log_dir+'/')

        inputs,loss_index,ground_true=batch
        output=self.model(
            input_ids=inputs,
            input_embeddings=None,
            attention_mask=None,
            attention_bias=None,
            past_key_values=None,
            use_cache=None,
            output_hidden_states=False,
        )
        output=output.logits
        loss=cross_entropy_loss(output,ground_true,loss_index)
        if self.trainer.is_global_zero:
            self.writer.add_scalar('Loss/train',loss,global_step=self.now_train_step)
            self.now_train_step=self.now_train_step+1
        return loss
    
    # def vali_step(self,batch,batch_idx):
    #     return None

    # def test_step(self,batch,batch_idx):
    #     return None



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.args['learn_rateing']),
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=0.1
        )
        return optimizer
