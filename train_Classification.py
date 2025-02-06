import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import math
import torch
import timm
import shutil
import logging
from typing import Sequence, Union
import multiprocessing as mp
import csv


import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, classification_report

import models.transformer_cust as ViT

from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from skimage.transform import resize 
from sklearn.model_selection import train_test_split


from torch.optim.lr_scheduler import _LRScheduler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

torch.set_num_threads(4) 

class MRIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        #self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def z_score_normalization(self, image):
        mean = np.mean(image)
        std = np.std(image)
        
        # Z-score normalization
        normalized_image = (image - mean) / std
        
        return normalized_image

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        # MRI 이미지 불러오기 (nii.gz 파일)
        image = nib.load(file_path).get_fdata()  # Numpy 배열로 변환
        image = np.expand_dims(image, axis=0)  # (채널, H, W, D) 형태로 확장
        image = self.z_score_normalization(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    # def __init__(self, set_idx, img, name,mode):
    #     self.img = img[set_idx,:]
    #     self.name = name[set_idx]
    #     self.mode = mode
        

    # def __len__(self):
    #     return len(self.name)

    # def z_score_normalization(self, image):
    #     mean = np.mean(image)
    #     std = np.std(image)
        
    #     # Z-score normalization
    #     normalized_image = (image - mean) / std
        
    #     return normalized_image
    
    # def __getitem__(self, idx):
    #     MRI_image = self.img[idx]
    #     MRI_image = self.z_score_normalization(MRI_image)
    #     name = self.name[idx]
    #     if name == 'CN':
    #         name = 0
    #     elif name == 'AD':
    #         name = 1

        # if self.mode == "train" :
        #     # p1 = np.random.uniform() # Mixup
        #     # if p1 <= 0.2 :
        #     #     mixup_value = np.random.uniform()
        #     #     while(True) :
        #     #         mixup_idx = random.randint(0, len(self.subjects)-1)
        #     #         if self.labels[mixup_idx] == label :
        #     #             mixup_path = PET_SRP_img_dir.format(site=self.sites[mixup_idx],subj=self.subjects[mixup_idx])
        #     #             mixup_img = nib.load(mixup_path).get_fdata()
        #     #             PET_image = PET_image * mixup_value + (1-mixup_value) * mixup_img
        #     #             break

        #     p2 = np.random.uniform() # flip
        #     if p2 <= 0.2 :

        #         np.flip(MRI_image, axis = 0)

        #     p3 = np.random.uniform() # rotate
        #     if p3 <= 0.2 :
        #         angles = [np.random.uniform(-0.1, 0.1) * 180 / np.pi for _ in range(3)]  # Convert radians to degrees
        #         MRI_image = rotate(MRI_image, angles[0], axes=(1, 2), reshape=False, mode='constant')  # Rotate around x-axis
        #         MRI_image = rotate(MRI_image, angles[1], axes=(0, 2), reshape=False, mode='constant')  # Rotate around y-axis
        #         MRI_image = rotate(MRI_image, angles[2], axes=(0, 1), reshape=False, mode='constant')  # Rotate around z-axis


        # MRI_image = torch.from_numpy(MRI_image)
        # MRI_image = torch.unsqueeze(MRI_image, 0)

        # return MRI_image, name

def data_file_reader(txt_file):
    tem_data = []
    with open(txt_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            file_path = row[0]  # 이미지 파일 경로
            label = int(row[1])  # 성별 (Male/Female)
            #label = 0 if gender_label == 'Dementia' else 1  # Dementia -> 1, NC -> 1
            tem_data.append((file_path, label))

    return tem_data

def data_loaders(data, mode, batch_size):

    # 데이터 읽기
    txt_file = args.data_dir  # 텍스트 파일 경로

    train_data=data_file_reader(os.path.join(txt_file,'ADNI_96_104_96_2DX_train.txt'))
    val_data=data_file_reader(os.path.join(txt_file,'ADNI_96_104_96_2DX_val.txt'))
    test_data=data_file_reader(os.path.join(txt_file,'ADNI_96_104_96_2DX_test.txt'))



    # 4. Dataset 및 DataLoader 생성
    train_dataset = MRIDataset(train_data)
    val_dataset = MRIDataset(val_data)
    test_dataset = MRIDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader,val_loader,test_loader

    # # Train-Valid 데이터 분할 (80% Train, 20% Valid)
    # train_idx, test_idx = train_test_split(np.arange(label.shape[0]),stratify=label, test_size=0.2, random_state=11)

    # if mode == 'Train_Test':
    #     # 각 데이터셋 생성
    #     train_dataset= DLB_Pretrain_dataset(train_idx,img,name, mode = 'train')
    #     test_dataset= DLB_Pretrain_dataset(test_idx,img,name, mode = 'val')

    #     #test_dataset = DLB_Pretrain_dataset (test_idx,mode = "test", augment_factor=1)

    #     # DataLoader 생성
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    #     #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    #     return train_loader, test_loader
    
    # elif mode == 'Train_Val_Test':

    #     train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=[label[idx] for idx in train_idx], random_state=11)
        
    #     train_dataset= DLB_Pretrain_dataset(train_idx,img,name, mode = 'train')
    #     val_dataset= DLB_Pretrain_dataset(val_idx,img,name, mode = 'val')
    #     test_dataset = DLB_Pretrain_dataset(test_idx,img,name,mode = "test")

    #     # DataLoader 생성
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    #     return train_loader, val_loader, test_loader


    

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def reduce_learning_rate(optimizer, iteration_count) :
    lr = args.lr - (args.lr * 1e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def performance(cm):
    accuracy = []
    sensitivity = []
    specificity = []
    total_tp = 0
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = cm.sum() - tp - fn - fp
        total_tp += tp
        class_total =np.sum(cm[:,i])
        
        accuracy.append(tp/class_total if class_total != 0 else 0)
        sensitivity.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0)

    total_acc = total_tp / cm.sum()

    return total_acc, accuracy, sensitivity, specificity

       
def Class_train_epoch(
    train_loader, val_loader, loss_fun, network, optimizer,scheduler, cur_epoch, args,K, writer=None
):
    #min_loss = math.inf
    global min_loss
    global max_acc
    global CM_v

    to_loss = 0
    network.train()
    print('lr : {}'.format(optimizer.param_groups[0]['lr']))

    train_predic=[]
    val_predic = []
    label_list= []
    for i, (MRI_images, labels) in enumerate(train_loader):

        MRI_images = MRI_images.float().to(device)
        labels = labels.float().type(torch.LongTensor).to(device)
  
        preds = network(MRI_images).float() #class 개수만큼으로 나눠 줌
        preds = preds.squeeze(1)

        loss = loss_fun(preds, labels)
        to_loss += loss.sum()

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.sum().backward()
         # Update the parameters.
        optimizer.step()
        scheduler.step()

        _,predic_class =torch.max(preds, dim=1)


        train_predic.append(predic_class.tolist())
        label_list.append(labels.tolist())

        del MRI_images, preds,#freeing gpu space
    train_loss = to_loss / (i+1) #최종 total loss / 배치 개수 

    train_predic = sum([sublist for sublist in train_predic],[])
    label_list = sum([sublist for sublist in label_list],[])
    #print(y_train)
    cf_t = confusion_matrix(label_list, train_predic)

    #Class 개수에 맞춰서 변경
    train_accuracy, accuracy, sensitivity, specificity = performance(cf_t)
    print(f'train_loss : {train_loss}, train_accuracy : {train_accuracy}')
    print(f'NC | acc : {accuracy[0]}, sensitivity :{sensitivity[0]}, specificity :{specificity[0]}')
    print(f'AD | acc : {accuracy[1]}, sensitivity :{sensitivity[1]}, specificity :{specificity[1]}')
    # print(f'EMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
    # print(f'LMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
    #print(f'Mixed | acc : {accuracy[3]}, sensitivity :{sensitivity[3]}, specificity :{specificity[3]}')

    
    val_predic=[]
    label_list = []
    to_loss = 0
    K=K
    network.eval()
    with torch.no_grad():
        for i, (MRI_images, labels) in enumerate(val_loader):
            # Compute the validation loss
                      
            MRI_images = MRI_images.float().to(device)
            labels = labels.type(torch.LongTensor).to(device)

            preds = network(MRI_images).float()   
            preds = preds.squeeze(1) 
         
            
            loss = loss_fun(preds, labels)
            to_loss += loss.sum()
            
            
            _,predic_class =torch.max(preds, dim=1)

            val_predic.append(predic_class.tolist())
            label_list.append(labels.tolist())

            del MRI_images, labels #freeing gpu space    
            
        val_loss = to_loss / (i+1)

        val_predic = sum([sublist for sublist in val_predic],[])
        label_list = sum([sublist for sublist in label_list],[])
        cf_v = confusion_matrix(label_list, val_predic)

        #class 개수에 맞춰서 변경
        val_accuracy, accuracy, sensitivity, specificity = performance(cf_v)

        print(f'val_loss : {val_loss}, val_accuracy : {val_accuracy}')
        print(f'NC | acc : {accuracy[0]}, sensitivity :{sensitivity[0]}, specificity :{specificity[0]}')
        print(f'AD | acc : {accuracy[1]}, sensitivity :{sensitivity[1]}, specificity :{specificity[1]}')
        # print(f'EMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
        # print(f'LMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
        # print(f'Mixed | acc : {accuracy[3]}, sensitivity :{sensitivity[3]}, specificity :{specificity[3]}')
                


        if max_acc <= val_accuracy :
            torch.save({
                'model_state_dict' : network.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()}, '{}/{}/weight_{}ep.tar'.format(log_dir, args.task, args.max_epoch))
            max_acc = val_accuracy
            K = cur_epoch
            CM_v = cf_v

    return train_loss, val_loss, train_accuracy, val_accuracy, val_predic ,K,CM_v


def Plot(train_loss_list, val_loss_list,train_acc_list, val_acc_list) :
    plt.figure(figsize=(40,20))
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.subplot(1, 2, 1)
    plt.title('Loss history', fontsize=30)
    plt.plot(torch.stack(train_loss_list).cpu().detach().numpy(), label='Train')
    plt.plot(torch.stack(val_loss_list).cpu().detach().numpy(), label='Validation')
    plt.xlabel('# of epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.legend(fontsize=30)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title('Accuracy history', fontsize=30)
    plt.plot(train_acc_list, label='Train')
    plt.plot(val_acc_list, label='Validation')
    plt.xlabel('# of epoch', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.legend(fontsize=30)
    plt.grid()
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick', labelsize=20)

    # plt.subplot(2, 2, 3)
    # plt.title('Sensitivity history', fontsize=30)
    # plt.plot(train_sen_list, label='Train')
    # plt.plot(val_sen_list, label='Validation')
    # plt.xlabel('# of epoch', fontsize=30)
    # plt.ylabel('Sensitivity', fontsize=30)
    # plt.legend(fontsize=30)
    # plt.grid()
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick', labelsize=20)

    # plt.subplot(2, 2, 4)
    # plt.title('Specificity history', fontsize=30)
    # plt.plot(train_spec_list, label='Train')
    # plt.plot(val_spec_list, label='Validation')
    # plt.xlabel('# of epoch', fontsize=30)
    # plt.ylabel('Specificity', fontsize=30)
    # plt.legend(fontsize=30)
    # plt.grid()
    # plt.rc('xtick',labelsize=20)
    # plt.rc('ytick', labelsize=20)

    plt.savefig('{}/{}/{}_train_val.png'.format(args.save_dir,args.task,args.task))

    print(f'End of Training')


parser = argparse.ArgumentParser()

# training options
parser.add_argument('--task', default='ADvsDLB',
                    help='Task')
parser.add_argument('--save_dir', default='/nasdata4/3_kmk/DLB/DLB_Transformer_MRI/output',
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--mask_ratio', type=float, default="0.75")
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--finetune', action="store_true")
parser.add_argument("--data_dir", default="/nasdata4/3_kmk/Dataset/ADNI_MRI/cohort0to3_MRI_dir.csv", help="dataset directory")
parser.add_argument('--image_size', type=int, nargs=3,default=(128,176,176))
parser.add_argument('--patch_size', type=int, nargs=3, default=(8,8,8))
parser.add_argument('--class_num', type=int, default=2)

args = parser.parse_args()

args.image_size = tuple(args.image_size)
args.patch_size = tuple(args.patch_size)






USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

log_dir = args.save_dir+'/logs'

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer_log_dir = '{}/{}'.format(log_dir,args.task)
if os.path.exists(writer_log_dir):
    shutil.rmtree(writer_log_dir)
writer = SummaryWriter(log_dir=writer_log_dir)


if not os.path.exists('{}/Task/{}/'.format(args.save_dir,args.task,args.task)):
    os.mkdir('{}/Task/{}/'.format(args.save_dir,args.task,args.task))



model = ViT.transformer_cust_classification_task(
    in_channels= 1,
    MRI_img_size=args.image_size,
    patch_size=args.patch_size,
    hidden_size=768,
    mlp_dim = 768*4,
    num_layers = 12,
    num_heads = 12,
    pos_embed = "conv",
    class_num = args.class_num,
)

 # pretrained weight load


#0.75
checkpoint = torch.load('./logs/ADNI_pretrained_making_50_96_104_96_NC_AC/weight_100ep.tar')  #validation loss가 가장 낮은

#0.5
#checkpoint = torch.load('./logs/MRI_cohort_96_104_MAE_pretraining_ViT_patchsize_8_masking_50_768_decoder_without_aug_22_re_7000ep_re_re/weight_7000ep.tar')  #validation loss가 가장 낮은
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'],strict=True)

model.to(device)


optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr, weight_decay = 0.05 )
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 3, T_mult=2, eta_min = 1e-8)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch : 0.75 ** epoch)
#T_0 : 반복하는 주기, T_mult : 주기 배수 , T_up : 초기 warmup, gamma : 주기마다 곱해지는 ratio, eta_max : 최대로 올라가는 LR  
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=2000, cycle_mult=2.0,max_lr=args.lr, min_lr = 1e-8, warmup_steps=400, gamma = 0.5)  

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=10, pct_start = 0.2, steps_per_epoch =10, max_lr =0.1)


loss_fun = nn.CrossEntropyLoss().to(device) #multi-class


train_loader, val_loader, test_loader = data_loaders(data = args.data_dir, mode='Train_Val_Test', batch_size=args.batch_size)


train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
train_sen_list = []
val_sen_list = []
train_spec_list =[]
val_spec_list =[]


start_epoch=0
K=0
max_acc=0
for cur_epoch in tqdm(range(start_epoch, args.max_epoch+1)):

    cur_epoch+=1

    train_loss, val_loss, train_accuracy, val_accuracy, val_predic, K, CM_v = Class_train_epoch(train_loader, val_loader, loss_fun, model, optimizer, scheduler, cur_epoch, args,K, writer) #writer = writer 

    writer.add_scalar('train_loss', train_loss, cur_epoch)
    writer.add_scalar('val_loss', val_loss, cur_epoch)
    writer.add_scalars('accuracy',{'train_accuracy' : train_accuracy, 'val_accuracy': val_accuracy},cur_epoch)


    for name,param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy())

    #train_loss_list.append(train_loss)
    #val_loss_list.append(val_loss)

    # train_acc_list.append(train_accuracy.item())
    # val_acc_list.append(val_accuracy.item())
    

    del train_loss, val_loss


            
print(f'The lowest loss : {K}_epoch')
disp_t = ConfusionMatrixDisplay(confusion_matrix=CM_v)
disp_t.plot(cmap=plt.cm.Blues)
plt.savefig('{}/Task/{}/best_val_acc_CM.png'.format(args.save_dir,args.task, args.task))

writer.close()


test_predic=[]
label_list = []
to_loss=0
loss_list=[]

checkpoint = torch.load('{}/{}/weight_{}ep.tar'.format(log_dir, args.task, args.max_epoch))  #validation loss가 가장 낮은
model.load_state_dict(checkpoint['model_state_dict'],strict=True)

with torch.no_grad():
    model.eval()
    for i, (MRI_images, labels) in tqdm(enumerate(test_loader)):
                # Compute the validation loss
        MRI_images = MRI_images.float().to(device)
        labels = labels.float().type(torch.LongTensor).to(device)

        preds = model(MRI_images).float()
        preds = preds.squeeze(1)

        _,predic_class =torch.max(preds, dim=1)
        test_predic.append(predic_class.tolist())
        label_list.append(labels.tolist())

        del MRI_images, labels,preds #freeing gpu space    


    test_predic = sum([sublist for sublist in test_predic],[])
    label_list = sum([sublist for sublist in label_list],[])
    cf_t = confusion_matrix(label_list, test_predic)
    print(f'test confusion matrix : {cf_t}')

    test_accuracy, accuracy, sensitivity, specificity = performance(cf_t)
    print(f'test_accuracy : {test_accuracy}')
    print(f'NC | acc : {accuracy[0]}, sensitivity :{sensitivity[0]}, specificity :{specificity[0]}')
    print(f'AD | acc : {accuracy[1]}, sensitivity :{sensitivity[1]}, specificity :{specificity[1]}')
    # print(f'EMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
    # print(f'LMCI | acc : {accuracy[2]}, sensitivity :{sensitivity[2]}, specificity :{specificity[2]}')
    #print(f'Mixed | acc : {accuracy[3]}, sensitivity :{sensitivity[3]}, specificity :{specificity[3]}')  
    print(classification_report(label_list, test_predic, target_names=["NC", "AD"],digits=5))

    disp_t = ConfusionMatrixDisplay(confusion_matrix=cf_t)
    disp_t.plot(cmap=plt.cm.Blues)


    plt.savefig('{}/Task/{}/Test.png'.format(args.save_dir,args.task,args.task))
