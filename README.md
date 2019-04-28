# TDT4265 - Edges2faces

##Getting started
- Clone this repo:
```bash
git clone https://github.com/SByfuglien/Edges2faces-pix2pix
cd Edges2faces-pix2pix
```
## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train
Download the dataset and the pretrained model from the following driver:

https://drive.google.com/drive/folders/1wovSAhRhwYH9UuWqsrXpP5GvcNRouIiX?usp=sharing

Run with vanilla:
```bash
python3 train.py --model Edges2faces-pix2pix --name edges2faces_vanilla --dataset_mode edges2faces --dataroot dataset-full
```

Run with WGAN
```bash
python3 train.py --model Edges2faces-pix2pix --name edges2faces_wgan --dataset_mode edges2faces --dataroot dataset-full
```

Now a visdom server will open at the address: localhost:8097

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
