import os

args_list = [
#     '--dataset_name mnist_m --hidden_dim 20 --max_epoch 100 --num_sp 75 --device cuda:0',
#     '--dataset_name mnist_m --hidden_dim 64 --max_epoch 100 --num_sp 75 --device cuda:0',
#     '--dataset_name mnist_m --hidden_dim 128 --max_epoch 100 --num_sp 75 --device cuda:1',
#     '--dataset_name mnist_m --hidden_dim 256 --max_epoch 100 --num_sp 75 --device cuda:1',
    
    
    '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --device cuda:0',
    '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --drn_k 6 --device cuda:0',
    '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --drn_k 8 --device cuda:0',
    
    '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --device cuda:1',
    '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --drk_k 6 --device cuda:1',
    '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --drk_k 8 --device cuda:1',
    
    '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 200 --lr 0.01 --device cuda:1',
    '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 200 --lr 0.01 --drn_k 6 --device cuda:1',
    
    '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 200 --lr 0.01 --device cuda:0',
    '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 200 --lr 0.01 --drn_k 6 --device cuda:0',
]

run_cmd = 'cd ~/SPGIE_thesis && python train.py'

for args in args_list:
    cmd = f'{run_cmd} {args}'
    os.system(cmd)

