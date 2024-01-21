import argparse
import os
import torch
from torch.utils.data import DataLoader

from SFGnet import SFGNet as Net

from utils import quantize
from data import get_eval_set
import time
import cv2

parser = argparse.ArgumentParser(description='SCNet')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--eval_data_dir', type=str, default='./data/test/input')
parser.add_argument('--eval_label_dir', type=str, default='./data/test/target')
parser.add_argument('--model', default='models/misTrain/net_epoch5.pth', help='Pretrained base model')
parser.add_argument('--save_dir', type=str, default='results1/')

opt = parser.parse_args()
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
test_set = get_eval_set(opt.eval_data_dir, opt.eval_label_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

model = Net(opt)
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')


if cuda:
    model = model.cuda(device)


def eval():

    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        with torch.no_grad():
            Input, Target, name = batch[0], batch[1], batch[2]
        if cuda:
            Input = Input.cuda(device)
        t0 = time.time()
        with torch.no_grad():
            
            out1,restored,m= model.forward(Input)
            
            prediction = quantize(restored, opt.rgb_range)
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            save_img(prediction.cpu().data, name[0])


def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])


eval()

