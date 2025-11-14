import torch
import torch.nn as nn
from transformers import AutoTokenizer
import copy
class process_data:
    def __init__(self,args):
        self.args=args
        self.tokenizer=AutoTokenizer.from_pretrained(self.args['default_checkpoint'])
        
    def get_output(self,inputs,max_length):
        # 这个函数要处理inputs，返回input_id,truth_id,ans_index
        data=[]
        temp={'prompt':'你是谁？','ans':'首先我是一个废物，其次我也是一个废物','mask':'','ans_index':[]}
        temp['prompt']='<|start_header_id|>user<|end_header_id|>\n\n' + temp['prompt'] + '<|eot_id|>'+'<|start_header_id|>assistant<|end_header_id|>\n\n'
        truth_id=self.tokenizer(temp['prompt']+temp['ans']+'')['input_ids']
        ans_index=[4,5,6]
        input_id=copy.deepcopy(truth_id)
        for i in range(len(ans_index)):
            input_id[ans_index[i]]=126336
        ans_index_temp=torch.zeros((len(truth_id)))
        ans_index_temp[ans_index]=1
        return input_id,truth_id,ans_index_temp

if __name__=='__main__':
    args={'default_checkpoint':"./checkpoint/model_checkpoint"}
    temp_class=process_data(args)
    a,b,c=temp_class.get_output(1111)
    
        

        