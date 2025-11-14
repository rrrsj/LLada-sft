import torch
import os
import torch.nn as nn
import yaml
from module.module import MyModule
import pytorch_lightning as L
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
from  pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from dataloader.dataloader import mydataloader
from functools import partial

def main(args):
    seed_everything(config['seed'])

    model_dir=f'./checkpoint/save_checkpoints/{str(args["save_time"])}'
    os.makedirs(model_dir,exist_ok=True)
    shutil.copy('./config/config.yaml',model_dir+'/config.yaml')

    model=MyModule(args)
    # block_cls=type(model.model.transformer['blocks'][0])
    # policy=partial(lambda_auto_wrap_policy,lambda_fn=lambda m:isinstance(m,(block_cls)))
    policy = {nn.Linear}
    now_strategy=FSDPStrategy(auto_wrap_policy=policy)


    callbacks=[ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=config['save_topk'],
        save_last=True,
        monitor=None,
        every_n_train_steps=args['save_step'],
        filename='epoch={epoch}-step={step}',    
    )]
    
    if args['accelerator']=='cpu':
        trainer=L.Trainer(
            strategy=now_strategy,
            callbacks=callbacks,
            logger=False,
            accelerator='cpu',
            accumulate_grad_batches=args['accumulate_grad_batches'],
            enable_checkpointing=args['enable_checkpointing'],
            max_epochs=args['epoch'],
            num_nodes=args['num_nodes'],
            precision=args['precision'],
            sync_batchnorm=True
            )
    else:
        trainer=L.Trainer(
            strategy=now_strategy,
            callbacks=callbacks,
            logger=False,
            accelerator='gpu',
            devices=args['num_gpus'],
            accumulate_grad_batches=args['accumulate_grad_batches'],
            enable_checkpointing=args['enable_checkpointing'],
            max_epochs=args['epoch'],
            num_nodes=args['num_nodes'],
            precision=args['precision'],
            sync_batchnorm=True
            )

    if str(args['load_checkpoint']) != 'None':
        checkpoint=torch.load(args['load_checkpoint'],map_location='cpu')
        state_dict=checkpoint['state_dict']
        state_dict={k.replace("_forward_module",''):v for k,v in state_dict.items()}
        _,_=model.load_state_dict(state_dict,strict=False)

    datamodule=mydataloader(args)
    trainer.fit(model,datamodule)
 

if __name__=='__main__':
    with open('./config/config.yaml','r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)