# Toward sufficient spatial-frequency interaction for gradient-aware underwater image enhancement  (ICASSP'24)

Chen Zhao, Weiling Cai, Chenyu Dong and Ziqi Zeng

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2309.04089)

<hr />

> **Abstract:** *Underwater images suffer from complex and diverse degradation,
which inevitably affects the performance of underwater visual tasks.
However, most existing learning-based underwater image enhancement (UIE) methods mainly restore such degradations in the spatial
domain, and rarely pay attention to the fourier frequency information. In this paper, we develop a novel UIE framework based on
spatial-frequency interaction and gradient maps, namely SFGNet,
which consists of two stages. Specifically, in the first stage, we
propose a dense spatial-frequency fusion network (DSFFNet),
mainly including our designed dense fourier fusion block and
dense spatial fusion block, achieving sufficient spatial-frequency
interaction by cross connections between these two blocks. In
the second stage, we propose a gradient-aware corrector (GAC)
to further enhance perceptual details and geometric structures of
images by gradient map. Experimental results on two real-world
underwater image datasets show that our approach can successfully enhance underwater images, and achieves competitive performance in visual quality improvement.* 
<hr />

## Network Architecture


## Installation and Data Preparation



## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. 

```
python train.py
```

## Testing

After preparing the testing data in ```data/test/``` directory. 


```
python test.py 
```





## Results

之后有空再上传预训练模型和可视化结果（谷歌云太难用了，之前弄过没弄好）。如果您喜欢我们的工作，SFGnet对您的研究能有帮助，小趴菜在这里恳请大大们引用一下~~谢谢各位路过的官人啦啦啦啦啦啦~

PS:把环境装好（非常基本非常简单），准备好数据（训练集和测试机），就可以直接运行，我在上传前已经测试过了，木有bug~~~~
PS:搞了一个晚上不知道怎么用git上床github，中间最终上传的时候一直报这个错误：fatal: unable to access 'https://github.com/zhihefang/SFGNet.git/': Failed to connect to github.com port 443 after 21076 ms: Couldn't connect to server.完全不知道怎么解决，如果有大佬知道怎么解决，留言一下（我之前上传过一次，不知道这次怎么就不行了，救命阿）。最终我是通过github网页一个个上传文件夹的。

另外，训练集的文件结构是/data/train,下面有两个子文件夹，分别是input和target。测试机是/data/test,跟训练集一样。




## Citation
If you use our work, please consider citing:

  
    @article{zhao2023toward,
     title={Toward Sufficient Spatial-Frequency Interaction for Gradient-aware Underwater Image Enhancement},
     author={Zhao, Chen and Cai, Weiling and Dong, Chenyu and Zeng, Ziqi},
     journal={arXiv preprint arXiv:2309.04089},
     year={2023}
    


## Contact
Should you have any questions, please contact 2518628273@qq.com
 

