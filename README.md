# TDT4265 - Edges2faces

##Getting started
- Clone this repo:
```bash
git clone https://github.com/SByfuglien/Edges2faces-pix2pix
cd Edges2faces-pix2pix
```
### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Download dataset and pretrained model
Download the dataset and the pretrained model from the following driver:

https://drive.google.com/drive/folders/1wovSAhRhwYH9UuWqsrXpP5GvcNRouIiX?usp=sharing

Create folder: /checkpoints

Run with vanilla:
```bash
python3 train.py --model Edges2faces-pix2pix --name edges2faces_vanilla --dataset_mode edges2faces --dataroot dataset-full
```

Run with WGAN
```bash
python3 train.py --model Edges2faces-pix2pix --name edges2faces_wgan --dataset_mode edges2faces --dataroot dataset-full
```

Now a visdom server will open at the address: localhost:8097

Results can be found in the checkpoints-folder

##Draw your own images
Create folders where your drawing and the result will be saves: /drawing and /results

```bash
python3  draw_app.py --model edges2faces_vanilla --dataset_mode single  --name pretrained_model --dataroot drawing 
```

###Instructions
Press mouse: draw
Scoll: change pencil width
Press scroll: delete all
ctrl + press mouse: eraser

## Acknowledgments
Basis for our project [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
