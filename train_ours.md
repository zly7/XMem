
#### 我们的训练肯定是跳过第2步，exp_id应该是tensorboard记录的路径
--stage 之后的是要训练的步骤

python -m torch.distributed.launch --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network saves/XMem-s01.pth


#### 在我这环境中用的是cv_pro

salloc -p gpulab02 -N1 -c8 --gres=gpu:1 -q gpulab02

#### 
下面这个脚本还必须在train.py前面，不然还真的没法
python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network saves/XMem-s0.pth


#### 现在vscode能debug ，是launch.json 的
11G 2080TI batch_size可以调整成2
python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network saves/XMem-s0.pth --num_workers 1 --s3_batch_size 2



#### 查看tesorboard
tensorboard --logdir=saves --bind_all