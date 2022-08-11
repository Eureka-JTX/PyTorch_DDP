# PyTorch_DDP
尝试入门PyTorch中的并行（DP）及分布式（DDP）处理机制和模型， 并学习相关开源项目。

由于手头的PC只有AMD的集显，所以具体多GPU的应用要等到之后进行。

## 单机多GPU启动指令
- 如果要使用```train_multi_gpu_using_launch.py```脚本，使用以下指令启动
- ```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpu_using_launch.py```
- 其中```nproc_per_node```为并行GPU的数量
- 如果要指定使用某几块GPU可使用如下指令，例如使用第1块和第4块GPU进行训练：
- ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py```
## 多机多GPU启动指令
- 在每台电脑上使用以下指令
- ```python -m torch.distributed.launch --nproc_per_node=8 --nnodes=机器数 --nodes_rank=第几台机器 --master_addr="主节点IP" --master_port="主节点端口号" --use_env train_multi_gpu_using_launch.py```
## 学习资料链接
### 官方文档
- torch.nn.DataParallel

   https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel
- torch.nn.DistributedDataParallel
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel

### 视频教程
- ResNet网络简介+迁移学习: https://www.bilibili.com/video/BV1T7411T7wa/?spm_id_from=333.788&vd_source=62633c01136853fd64ea0ef64e3cc1a0
- 利用PyTorch实现ResNet: https://www.bilibili.com/video/BV14E411H7Uw/?spm_id_from=333.788&vd_source=62633c01136853fd64ea0ef64e3cc1a0 

   源码：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test5_resnet
- 利用PyTorch进行多GPU训练: https://www.bilibili.com/video/BV1yt4y1e7sZ?spm_id_from=333.999.0.0&vd_source=62633c01136853fd64ea0ef64e3cc1a0