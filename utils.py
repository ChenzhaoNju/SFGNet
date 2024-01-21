from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import numpy as np
from collections import OrderedDict
import glob
import torch
import cv2
import re
import os
import torch
from torch import sqrt,erf,exp,sign
import math
from natsort import natsorted


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def normalize(data):
    return data / 255.

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def calc_psnr(img1, img2):
    '''
        Calculate PSNR. img1 and img2 should be torch.Tensor and ranges within [0, 1].
    '''
    return
def batch_PSNR(img, imclean, data_range):
    Img = img.data.detach().cpu().numpy().astype(np.float32)
    Iclean = imclean.data.detach().cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])
def batch_SSIM(img, imclean,data_range):
    Img = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    Iclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SSIM = []
    #print(Iclean.shape)
    for i in range(Img.shape[0]):
        # ssim = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], gaussian_weights=True, use_sample_covariance=False, multichannel =True)
        ssim = structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:],  data_range=data_range, multichannel = True,win_size=5)
        #ssim = structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:], win_size=3,  data_range=data_range)
        #ssim = structural_similarity(Iclean[i,:,:,:], Img[i,:,:,:],  data_range=data_range)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM)
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*net_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch
def cal_psnr(x_, x):
    mse = ((x_.astype(np.float)-x.astype(np.float))**2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img
def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
def rizer_filters(size,scal,factor):
    #size: filter size
    #scal:Scale factor of multi-scale decomposition
    #factor: 一般取（1/2）^n (n = 1,2...)
    #f = exp(-w^2*q^2)/2
    #g = sqrt(1 - f**2)
    deter = factor * 2 ** (scal - 1)
    deter = torch.tensor(deter).cuda()
    w = []
    for i in range(-math.floor(size / 2), math.ceil(size / 2)):
        w.append(i)
    w = torch.tensor(w).cuda()
    #w的取值范围是（-pi,pi）
    pi=3.1415926
    w = w*2*pi/size
    w = torch.fft.ifftshift(w)
    length = len(w)
    w = w.reshape(length,1)

    f = exp(-(w**2)*(deter**2)/2)
    g = sqrt(1 - f**2)
    gh =  sign(w) * g * (-1j)


    f = f[:,0]
    g = g[:,0]
    gh = gh[:,0]
    return f,g,gh

