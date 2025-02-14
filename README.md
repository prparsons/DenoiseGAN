# DenoiseGAN
This is mainly just to satisfy my own curiousity, how well could a super resolution network be adapted for de-noising an image? Seems intuitive that they're generally looking at the same thing in an image, and more or less performing a similar task.

I left the generator pretty close to the original, though there are a few changes. In place of the upsampling/pixel shuffle layers, I used a depthwise separable convolution after the residual is added. Also reduced the size of the input/ouput layer kernels. At present, I haven't done any testing regarding these changes, only verified that the model does work.
The discriminator was mainly made much smaller (easier to get things training, will adjust that later also).
Made changes in the dataloader, obviously aren't scaling images. So it now is adding noise to the images, taking random crop for the training.  
Also made some changes in the loss function, didn't take much tinkering to get it to train decently well with a good loss profile for the discriminator. Biggest change in the loss calculation is the replacement of vgg for the perceptual loss with ResNet50. Mainly needed something smaller to train at home with limited RAM; wanted to use EfficientNet, don't have new enough versions of torchvision installed at home for that though (I've tested using that for SRGAN in place of VGG with pretty good results).  
Need to do some testing to see how well the perceptual loss affects this: does it help add/retain detail like it does for super resolution? does the layer choice (deeper/shallower) affect quality of fine detail like it does with super resolution?

It has worked decently well, enough so that I figured it was worth sharing.

Based on a PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```



## Usage

### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
```
The output val super resolution images are on `training_results` directory.

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution images are on `benchmark_results` directory.

### Test Single Image
```
python test_image.py

optional arguments:
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.

### Test Single Video
```
python test_video.py

optional arguments:
--video_name                  test low resolution video name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution video and compared video are on the same directory.


## Results

PSNR: 32.0438 dB SSIM: 0.9466

Image zoomed 200% to show detail<br>
Left: original<br>
Middle: noise added<br>
Right: De-noised

![Results image](https://github.com/prparsons/DenoiseGAN/blob/master/epoch_100_index_2.png)


Used the DIV2K training set for training (pulled a portion of those for validation).
http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf
@InProceedings{Ignatov_2018_ECCV_Workshops,
author = {Ignatov, Andrey and Timofte, Radu and others},
title = {PIRM challenge on perceptual image enhancement on smartphones: report},
booktitle = {European Conference on Computer Vision (ECCV) Workshops},
month = {January},
year = {2019}
}
