import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import math
import torch
import timm
import shutil
import logging
import multiprocessing as mps
import csv

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

import models.transformer_cust as ViT
from sklearn.model_selection import train_test_split

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
    


def data_loaders(data, mode, batch_size):

    txt_file = data  # 텍스트 파일 경로
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            file_path = line[0]  # 이미지 파일 경로
            gender_label = line[3].strip()  # 성별 (Male/Female)
            label = 0 if gender_label == 'Male' else 1  # Male -> 0, Female -> 1
            data.append((file_path, label))

    # DataFrame 생성
    df = pd.DataFrame(data, columns=['file_path', 'label'])

    # 3. Stratified Sampling을 이용한 데이터셋 분할
    train_data, val_data = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    # 리스트로 변환
    train_data = train_data.values.tolist()
    val_data = val_data.values.tolist()


    # 4. Dataset 및 DataLoader 생성
    train_dataset = MRIDataset(train_data)
    val_dataset = MRIDataset(val_data)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 

    return train_loader,val_loader,
    
    

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



def Predic_train_epoch(
    Predic_loader, test_loader, loss_fun, network, optimizer,scheduler, cur_epoch, args, K, writer=None
):

    global min_loss
    to_loss = 0

    network.train()

    print('lr : {}'.format(optimizer.param_groups[0]['lr']))


    for i, (MRI_images, labels) in enumerate(Predic_loader):

        MRI_images = MRI_images.float().to(device)

        pred_pixel_values_mask, pred_pixel_values_unmask, patches, batch_range, masked_indices,unmasked_indices,rearranged_img = network(MRI_images)
        loss = loss_fun(pred_pixel_values_mask, patches[batch_range, masked_indices])
        # loss2 = loss_fun(pred_pixel_values_unmask,patches[batch_range, unmasked_indices])
        # loss = 0.8*loss1.mean() + 0.2*loss2.mean()

        to_loss += loss.item()
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        

        del pred_pixel_values_mask, pred_pixel_values_unmask,batch_range, masked_indices,unmasked_indices, patches
    train_loss = to_loss/ (i+1) # 묶음 별로 mean loss 가져오는거라 총 batch 개수로 나눠줌


    print(f'train_loss : {train_loss}')

    if cur_epoch%50 == 0:
        recon_pic(rearranged_img, MRI_images, cur_epoch, i, mode='train')

    del MRI_images, rearranged_img
    

    K=K
    to_loss = 0
    network.eval()
    with torch.no_grad():
        for i, (MRI_images, labels) in enumerate(test_loader):
            # Compute the validation loss
                        
            MRI_images = MRI_images.float().to(device)


            pred_pixel_values_mask, pred_pixel_values_unmask, patches, batch_range, masked_indices,unmasked_indices,rearranged_img = network(MRI_images)
            loss = loss_fun(pred_pixel_values_mask, patches[batch_range, masked_indices])
            # loss2 = loss_fun(pred_pixel_values_unmask,patches[batch_range, unmasked_indices])
            # loss = 0.8*loss1.mean() + 0.2*loss2.mean()

            to_loss += loss.item()
            
            del labels,pred_pixel_values_mask, pred_pixel_values_unmask,batch_range, masked_indices,unmasked_indices, patches #freeing gpu space    

        if cur_epoch%50 == 0:
            recon_pic(rearranged_img, MRI_images, cur_epoch, i, mode='val')

        del MRI_images, rearranged_img

        val_loss = to_loss/ (i+1) 

        print(f'val_loss : {val_loss}')

        if min_loss > val_loss :
            torch.save({
                'model_state_dict' : network.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()}, '{}/{}/weight_{}ep.tar'.format(args.log_dir, args.task, args.max_epoch))
            min_loss = val_loss
            K = cur_epoch
        
        plt.close('all')


    return train_loss, val_loss , K



def Plot(train_loss_list, val_loss_list) :
        
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


    plt.savefig('{}/{}/{}_train_val.png'.format(args.save_dir,args.task,args.task))

    print(f'End of Training')


def recon_pic(preds, targets,epoch,j,mode):
    
    #Size 바뀔 때 어디 slice 잘라서 볼 건지 수정!
    # whole 3D, Slice, select Slice 
    preds = preds.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Slice20_Targets_{}'.format(mode), fontsize=20)
    plt.imshow(targets[0, :, :, :, 50].squeeze())

    plt.subplot(1, 2, 2)
    plt.title('Slice20_Preds_{}'.format(mode), fontsize=20)
    plt.imshow(preds[0, :, :, :, 50].squeeze())
    
    plt.savefig('{}/reconst/{}/Epoch_{}_{}.png'.format(args.save_dir,args.task,epoch,mode))

    del preds,targets
        


parser = argparse.ArgumentParser()       
# training options
parser.add_argument('--task', default='ADvsDLB',
                    help='Task')
parser.add_argument('--save_dir', default='/nasdata4/3_kmk/DLB/DLB_Transformer_MRI/output',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='/nasdata4/3_kmk/DLB/DLB_Transformer_MRI/output/logs',
                    help='Directory to save the log')
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
parser.add_argument("--data_dir", default="/nasdata4/3_kmk/Dataset/ADNI_MRI/cohort0to3_MRI_dir.csv", type=str, help="dataset directory")
parser.add_argument('--image_size', type=int, nargs=3,default=(128,176,176))
parser.add_argument('--patch_size', type=int, nargs=3, default=(8,8,8))
parser.add_argument('--class_num', type=int, default=2)

args = parser.parse_args()

args.image_size = tuple(args.image_size)
args.patch_size = tuple(args.patch_size)






USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


if not os.path.exists('{}/reconst/{}'.format(args.save_dir,args.task)):
    os.makedirs('{}/reconst/{}'.format(args.save_dir,args.task))

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

writer_log_dir = '{}/{}'.format(args.log_dir,args.task)
if os.path.exists(writer_log_dir):
    shutil.rmtree(writer_log_dir)
writer = SummaryWriter(log_dir=writer_log_dir)



model = ViT.transformer_cust(
    in_channels= 1,
    img_size=args.image_size,
    patch_size=args.patch_size,
    hidden_size=768,
    mlp_dim = 768*4,
    num_layers = 12,
    num_heads = 12,
    pos_embed = "conv",
    dropout_rate = 0.0,
    decoder_dim = 768,
    decoder_depth = 1,
    decoder_heads = 8,
    masking_ratio = args.mask_ratio,
)



# pretrained weight load

#checkpoint = torch.load('/nasdata4/3_kmk/DLB/DLB_Transformer_MRI/output/logs/MRI_cohort_96_104_MAE_pretraining_ViT_patchsize_8_masking_50_768_decoder_without_aug_22_add/weight_1000ep.tar')  #validation loss가 가장 낮은
model = nn.DataParallel(model)
#model.load_state_dict(checkpoint['model_state_dict'],strict=False)

model.to(device)
 



optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr, weight_decay = 0.05 )
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 3, T_mult=2, eta_min = 1e-8)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch : 0.75 ** epoch)
#T_0 : 반복하는 주기, T_mult : 주기 배수 , T_up : 초기 warmup, gamma : 주기마다 곱해지는 ratio, eta_max : 최대로 올라가는 LR  
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=4000, cycle_mult=2.0,max_lr=args.lr, min_lr = 1e-7, warmup_steps=400, gamma = 0.5)  

loss_fun = torch.nn.MSELoss().to(device) #prediction loss


img,name = np.load('/nasdata4/3_kmk/Dataset/Cohort/96_104_96_img.npy'), np.load('/nasdata4/3_kmk/Dataset/Cohort/cohort0to3_MRI_age.npy')

train_loader, val_loader = data_loaders(img, name, mode='Train_Test', batch_size=args.batch_size)

#start
    
min_loss = math.inf
max_acc = 0
CM_v = 0

train_loss_list = []
val_loss_list = []


start_epoch=0
K=0
for cur_epoch in tqdm(range(start_epoch, args.max_epoch+1)):

    cur_epoch+=1
    train_loss,val_loss,K= Predic_train_epoch(train_loader, val_loader, loss_fun, model, optimizer,scheduler, cur_epoch, args,K, writer) #writer = writer 

    writer.add_scalar('train_loss', train_loss, cur_epoch)
    writer.add_scalar('val_loss', val_loss, cur_epoch)

    for name,param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy())

    #train_loss_list.append(train_loss)
    #val_loss_list.append(val_loss)

    del train_loss, val_loss

#Plot(train_loss_list, val_loss_list)
print(f'The lowest loss : {K}_epoch')
writer.close()

