import random
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import argparse
from torch import nn
from SFGnet import SFGNet,Vgg,FFTLoss,Get_gradient,AFFTLoss
import torch.optim as optim
from torch.autograd import Variable
from utils import findLastCheckpoint,batch_PSNR,batch_SSIM
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data_RGB import get_training_data, get_validation_data
from SFGnet import msssim
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_args():
 
    parser = argparse.ArgumentParser(description="MDARNet_train")
    parser.add_argument("--batchSize", type=int,required=False, default=5,help="Training batch size")
    parser.add_argument("--pachSize", type=int,required=False, default=256,help="Training batch size")
    parser.add_argument("--epochs", type=int, required=False , default=1600, help="Number of training epochs")   
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--save_weights", type=str, required=False,default="./models/misTrain", help='path of log files')
    parser.add_argument("--train_data", type=str, required=False, default='./data/train/', help='path to training data')
    parser.add_argument("--val_data", type=str, required=False, default='./data/test/', help='path to training data')
    parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    parser.add_argument('--decay', type=int, default='25', help='learning rate decay type')
    return parser.parse_args()

if __name__ == '__main__':
    opt = get_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # loading datasets	
    train_dataset = get_training_data(opt.train_data, {'patch_size': opt.pachSize})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4,drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(opt.val_data, {'patch_size': opt.pachSize})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)

    # loading model
    H = opt.pachSize
    W = opt.pachSize
    model = SFGNet(H=H,W=W,batch_size=opt.batchSize)

    ######### Loss ###########
    criterion = SSIM()
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    

    vgg=Vgg()
    AFFT=AFFTLoss()
    FFT=FFTLoss()
    msssim_loss = msssim

    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
        criterion1.cuda()
        FFT.cuda()
        AFFT.cuda()
   
    # Optimizer
    milestones = []
    for i in range(1, opt.epochs+1):
        if i % opt.decay == 0:
          milestones.append(i)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9, 0.999),eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)  # learning rates
    warmup_epochs = 4
    scheduler.step()
    initial_epoch = findLastCheckpoint(save_dir=opt.save_weights)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_weights, 'net_epoch%d.pth' % initial_epoch)))
    best_psnr = 0
    best_ssim = 0
    signal = True
    for epoch in range(initial_epoch, opt.epochs):
        print("======================",epoch,'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_lossg = 0
        if signal:
                lossg = 0.1
                signal = False
        else:
                lossg = float(lossgg)
        model.train()
        for i, (data) in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad()
            target = data [0].cuda()
            input_ = data [1].cuda()
            if opt.use_GPU:
                input_train, target_train = Variable(input_.cuda()), Variable(target.cuda())
            gradient = Get_gradient()
            grad_gt=gradient(target)
            out,out2,g = model(input_,grad_gt)
            lossg= criterion1(g, grad_gt)

            msssim_loss_ = 1-msssim_loss(target, out2, normalize=True)
            FFT1=AFFT(target, out)
            FFT2=FFT(target, out2)
            loss11 = criterion1(target, out)
            loss1 = criterion1(target, out2)
            va_vgg = vgg(out2)
            vb_vgg = vgg(out)
            vp_vgg = vgg(target)
            vn_vgg = vgg(input_)
            cl_loss = 0
            weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            for i in range(len(va_vgg)):
              d_ap = criterion1(va_vgg[i], vp_vgg[i].detach())
              d_an = criterion1(va_vgg[i], vn_vgg[i].detach())
              d_ab = criterion1(va_vgg[i], vb_vgg[i].detach())
              contra = d_ap / (d_an + d_ab+  1e-7)
              cl_loss += weights[i] * contra

            losss1 = FFT1
            losss2 = msssim_loss_+FFT2+cl_loss+loss1
            loss = 0.1*losss1+losss2+0.5*lossg
            loss.backward()
            optimizer.step()
            print("[epoch %d] f  loss: %.4f  lossg: %.4f "\
                  % (epoch + 1,  
                     loss, lossg))
            epoch_loss += loss.item()
            epoch_lossg += lossg.item()
        scheduler.step()
        model.eval()
        psnr_val_rgb = 0
        ssim_val_rgb=0
        psnr_val_rgb_out1=0
        ssim_val_rgb_out1=0
        gradient = Get_gradient()
            
       
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val [0].cuda()
            input_ = data_val [1].cuda()
            if opt.use_GPU:
                target = target.cuda()
                input_ = input_.cuda()
            

            with torch.no_grad():
                gt=gradient(target)
                out1, restored,g = model(input_)

            out1 = torch.clamp(out1, 0., 1.)
            psnr_train_out1 = batch_PSNR(out1, target, 1.)
            ssim_train_out1 = batch_SSIM(out1, target, 1)
            psnr_val_rgb_out1+=psnr_train_out1
            ssim_val_rgb_out1+=ssim_train_out1

            restored = torch.clamp(restored, 0., 1.)
            psnr_train = batch_PSNR(restored, target, 1.)
            ssim_train = batch_SSIM(restored, target, 1)
            psnr_val_rgb+=psnr_train
            ssim_val_rgb+=ssim_train
        epoch_psnr_out1 = psnr_val_rgb_out1/len(val_loader)
        epoch_ssim_out1 = ssim_val_rgb_out1/len(val_loader)

            
        epoch_psnr = psnr_val_rgb/len(val_loader)
        epoch_ssim = ssim_val_rgb/len(val_loader)
        if best_psnr<epoch_psnr:
            best_psnr = epoch_psnr
        if best_ssim<epoch_ssim:
            best_ssim = epoch_ssim
        #lossgg=0
        #lossgg==epoch_lossg / len(train_loader)
        print("=========epoch:%d"%epoch,"========epoch PSNR:%f"%epoch_psnr,"========best PSNR:%f"%best_psnr,"========best SSIM:%f"%best_ssim)
        torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_epoch%d.pth' % (epoch+1)))
        torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_last.pth'))
        f = open('./Train.txt', mode='a')
        f.write('epoch:' + '%2.4f' % (epoch + 1) + '    ')
        f.write('lr={:.6f}'.format(scheduler.get_lr()[0]) + '    ')
        f.write('epoch_loss:' + '%2.4f' % (epoch_loss / len(train_loader)) + '    ')
        f.write('epoch_out1ssim:' + '%2.4f' % epoch_ssim_out1 + '       ')
        f.write('epoch_out1psnr:' + '%2.4f' % epoch_psnr_out1 + '\n')
        f.write('epoch_ssim:' + '%2.4f' % epoch_ssim + '       ')
        f.write('epoch_psnr:' + '%2.4f' % epoch_psnr + '\n')
        f.close()






