import torch
from torch.utils.data import Dataset
import os
import random
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname('../')))
from utils.process_dataset_data import process_data
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self,args,rank_id,global_id,data_type='train'):
        super().__init__()
        self.global_id=global_id
        self.rank_id=rank_id
        self.args=args
        self.data_type=data_type
        self.file_buffer=self.args['file_buffer_size']
        self.max_length=self.args['train_max_length']
        self.process=process_data(args)
        self.now_file_point=0
        self.now_data_point=1
        self.get_file_list()
        
    def __len__(self):
        return self.args['train_step']

    def get_file_list(self):
        data_path=self.args['data_path']+'/'+self.data_type
        files = [data_path+'/'+f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        random.shuffle(files)
        self.files_now_dataset=[files[i*self.global_id+self.rank_id] for i in range((len(files)+self.global_id-self.rank_id-1)//self.global_id)]
        self.data_buffer=[]
    
    def fill_buffer(self):
        self.data_buffer=[]
        self.now_data_point=0
        for i in range(self.file_buffer):
            with open(self.files_now_dataset[self.now_file_point%len(self.files_now_dataset)],'r') as f:
                while True:
                    now_data = f.readline()
                    print(now_data)
                    if not now_data:  # 如果读到空行（文件末尾）
                        break
                    # now_data 处理成prompt(question), input(step{i}), answer(step{i+1})
                    data_temp=json.loads(now_data)
                    for key in data_temp.keys():
                        if 'step' in key:
                            try:
                                now_step=int(key[4:])
                                now_temp={}
                                now_temp['question']=data_temp['question']
                                now_temp['structured_answer']=data_temp['structured_answer']
                                now_temp['now_step']=data_temp[key]
                                now_temp['next_step']=data_temp['step'+str(now_step+1)]
                                self.data_buffer.append(now_temp)
                            except:
                                now_temp={}
                                now_temp['question']=data_temp['question']
                                now_temp['structured_answer']=data_temp['structured_answer']
                                now_temp['now_step']='[MASK]'
                                now_temp['next_step']=data_temp['step1']
                                self.data_buffer.append(now_temp)
            self.now_file_point=self.now_file_point+1
        random.shuffle(self.data_buffer)
    
    def __getitem__(self,data_index):
        if self.args['use_casual_document_mask'] and self.data_type=='train':
            attention_mask=torch.zeros((self.max_length,self.max_length))
            now_length=0
            now_input=[]
            now_output=[]
            all_ans_index=[]
            attention_mask=[]
            sentence_number=0
            while now_length<self.max_length:
                if self.now_data_point>=len(self.data_buffer):
                    self.fill_buffer()
                max_length=self.max_length
                input_id,truth_id,ans_index=self.process.get_output(self.data_buffer[self.now_data_point],max_length)
                for i in range(len(ans_index)):
                    all_ans_index.append(ans_index[i]+now_length)
                attention_mask=attention_mask+[sentence_number for i in range(len(input_id))]
                now_input=now_input+input_id
                now_output=now_output+truth_id
                now_length=now_length+len(input_id)
                self.now_data_point=self.now_data_point+1
            return torch.tensor(now_input),torch.tensor(now_output),torch.tensor(attention_mask),all_ans_index

        if self.data_type=='train' and self.args['use_casual_document_mask']==False:
            if self.now_data_point>=len(self.data_buffer):
                self.fill_buffer()
            max_length=self.max_length  
            # get_output: input_id(step{i}), truth_id(step{i+1}), ans_index(要算哪些token的loss)
            input_id,truth_id,ans_index=self.process.get_output(self.data_buffer[self.now_data_point],max_length)
            assert len(input_id)==len(truth_id)
            self.now_data_point=self.now_data_point+1
            return torch.tensor(input_id),torch.tensor(truth_id),ans_index
        elif self.data_type=='vali':
            return 0

if __name__=='__main__':
    args={'file_buffer_size':1,'train_max_length':1024,'data_path':'./data/generate_data','default_checkpoint':"/home/share/LLaDA-1.5/",'train_step':1000,'use_casual_document_mask':False}
    dataset=CustomDataset(args,rank_id=0,global_id=1,data_type='train')
    dataset.fill_buffer()
    print(dataset.data_buffer[0])
        