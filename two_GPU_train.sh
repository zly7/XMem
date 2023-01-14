#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab02      # 作业提交的指定分区队列为titan
#SBATCH --qos=gpulab02           # 指定作业的QOS
#SBATCH -J 2GPU_train       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1，节点相当于机器
#SBATCH --ntasks-per-node=12    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

python -m torch.distributed.run --master_port 25764 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network saves/XMem-s0.pth --num_workers 1 --s3_batch_size 2