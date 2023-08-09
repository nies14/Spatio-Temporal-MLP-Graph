# Regular Splitting Graph Network for 3D Human Pose Estimation (RS-Net)
<p align="center"><img src="./demo/Network_Architecture.png", width="600" alt="" /></p>
The PyTorch implementation for RS-Net.

## Qualitative and quantitative results
<p align="center"><img src="demo/squat.gif", width="400"  alt="" /></p>

| Method | MPJPE(mm) | PA-MPJPE(mm) |
|  :----:  | :----: | :----: |
| [SemGCN](https://github.com/garyzhao/SemGCN) | 57.6 | - |
| [High-order GCN](https://github.com/ZhimingZo/HGCN) | 55.6 | 43.7 |
| [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation) | 54.8 | 42.9 |
| [Weight Unsharing](https://github.com/tamasino52/Any-GCN) | 52.4 | 41.2 |
| [ModulatedGCN](https://github.com/ZhimingZo/Modulated-GCN) | 49.4 | 39.1 |
| Ours | **47.0** | **38.6** |

## Dependencies

Make sure you have the following dependencies installed:

* PyTorch >= 1.7.0
* NumPy
* Matplotlib
* FFmpeg (if you want to export MP4 videos)
* ImageMagick (if you want to export GIFs)

## You can create the environment:
```bash
conda create -n rsnet python=3.8
conda activate rsnet
pip install -r requirements.txt
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M & MPI-INF-3DHP
We set up the Human3.6M & MPI-INF-3DHP dataset in the same way as [PoseAug](https://github.com/jfzhang95/PoseAug). Please refer to [DATASETS.md](https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md) for the preparation of the dataset files & put them in `./dataset` directory.


## Evaluating our models
You can download our pre-trained models from [here](https://drive.google.com/drive/folders/1gWk1B-q-220XR-9MqdlqJFtUI3eBVJe6?usp=sharing). Put them in the `./checkpoint` directory.
### Human 3.6M

To evaluate our pre-trained model using the detected 2D keypoints (HR-Net) with pose refinement, please run:
```bash
python main_graph.py -k hr --post_refine --rsnet_reload 1 --post_refine_reload 1 --save_out_type post --show_protocol2 --previous_dir './checkpoint/HR-Net' --rsnet_model model_rsnet_2_eva_post_4704.pth --post_refine_model model_post_refine_2_eva_post_4704.pth --nepoch 2 -z 96 --batchSize 512
```

To evaluate our pre-trained model using ground truth 2D keypoints without pose refinement, please run:
```bash
python main_graph.py -k gt --post_refine --rsnet_reload 1 --show_protocol2 --previous_dir './checkpoint/GT' --rsnet_model model_rsnet_5_eva_xyz_3728' --nepoch 2 -z 64 --batchSize 128
```

## Training from scratch
### Human 3.6M

To train our model using the detected 2D keypoints (HR-Net) with pose refinement, please run:
```bash
python main_graph.py -k hr --pro_train 1 --save_model 1  --save_dir './checkpoint' --show_protocol2  --post_refine --save_out_type post -z 96 --batchSize 512 --nepoch 31
```

To evaluate our model using the detected 2D keypoints (HR-Net) with pose refinement, please run:
```bash
python main_graph.py -k hr --post_refine --rsnet_reload 1 --post_refine_reload 1 --save_out_type post --show_protocol2 --previous_dir './checkpoint/HR-Net' --rsnet_model '[model_rsnet]' --post_refine_model '[model_post_refine]' --nepoch 2 -z 96 --batchSize 512
```

To train our model on the ground truth 2D keypoints without pose refinement, please run:
```bash
python main_graph.py -k gt  --pro_train 1 --save_model 1  --save_dir './checkpoint/GT' --show_protocol2  -z 64 --batchSize 128 --nepoch 31 --learning_rate 1e-3 --large_decay_epoch 5 --lr_decay .95
```

To evaluate our model using ground truth 2D keypoints without pose refinement, please run:
```bash
python main_graph.py -k gt --rsnet_reload 1 --show_protocol2 --previous_dir './checkpoint/GT' --rsnet_model '[model_rsnet]' --nepoch 2 -z 64 --batchSize 128
```

## Acknowledgement
Our code refers to the following repositories.
* [ModulatedGCN](https://github.com/ZhimingZo/Modulated-GCN)
* [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation)
* [PoseAug](https://github.com/jfzhang95/PoseAug)
* [MHFormer](https://github.com/Vegetebird/MHFormer)

We thank the authors for releasing their codes.
