import torch
import torch.nn 
import torch.nn.functional as F

def cross_entropy_loss(prediction: torch.Tensor,ground_true:torch.Tensor,prediction_index: torch.Tensor):
    '''
        prediction batch,token,num_class
        ground_true batch,token
        prediction_index batch,prediction_token
    '''
    B,T,C=prediction.shape

    prediction_mask=prediction_index.reshape(-1,1)
    prediction=prediction.reshape(-1,C)
    ground_true=ground_true.reshape(-1,1).long()
    log_probs=torch.log_softmax(prediction,dim=-1)
    ce_loss=torch.gather(log_probs,dim=-1,index=ground_true)
    ce_loss=ce_loss*prediction_mask
    return (-ce_loss).mean()
    

    