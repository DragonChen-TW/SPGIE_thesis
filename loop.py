import os

args_list = [
# #     ==================== mnist ====================
#     '--dataset_name mnist --hidden_dim 20 --max_epoch 100 --device cuda:0',
#     '--dataset_name mnist --hidden_dim 64 --max_epoch 100 --device cuda:0',
#     '--dataset_name mnist --hidden_dim 128 --max_epoch 200 --device cuda:0',
#     '--dataset_name mnist --hidden_dim 256 --max_epoch 200 --device cuda:0',
    
# #     ==================== fashion_mnist ====================
#     '--dataset_name fashion_mnist --hidden_dim 20 --max_epoch 100 --device cuda:1',
#     '--dataset_name fashion_mnist --hidden_dim 64 --max_epoch 100 --device cuda:1',
#     '--dataset_name fashion_mnist --hidden_dim 128 --max_epoch 100 --device cuda:1',
#     '--dataset_name fashion_mnist --hidden_dim 256 --max_epoch 100 --device cuda:1',
    
# #     ==================== cifar10 ====================
#     '--dataset_name cifar10 --hidden_dim 20 --max_epoch 100 --device cuda:0',
#     '--dataset_name cifar10 --hidden_dim 64 --max_epoch 100 --device cuda:0',
#     '--dataset_name cifar10 --hidden_dim 128 --max_epoch 100 --device cuda:0',
#     '--dataset_name cifar10 --hidden_dim 256 --max_epoch 100 --device cuda:0',
    
#     ==================== mnist_m ====================
    # '--dataset_name mnist_m --hidden_dim 20 --max_epoch 100 --num_sp 75 --device cuda:0',
    # '--dataset_name mnist_m --hidden_dim 64 --max_epoch 100 --num_sp 75 --device cuda:1',
    # '--dataset_name mnist_m --hidden_dim 128 --max_epoch 100 --num_sp 75 --device cuda:1',
    # '--dataset_name mnist_m --hidden_dim 256 --max_epoch 100 --num_sp 75 --device cuda:0',

#     ==================== mnist_m ====================
    '--dataset_name mnist_m --hidden_dim 20 --max_epoch 200 --num_sp 75 --device cuda:0 --with_std True',
    '--dataset_name mnist_m --hidden_dim 64 --max_epoch 200 --num_sp 75 --device cuda:0 --with_std True',
    '--dataset_name mnist_m --hidden_dim 128 --max_epoch 200 --num_sp 75 --device cuda:0 --with_std True',
    '--dataset_name mnist_m --hidden_dim 256 --max_epoch 200 --num_sp 75 --device cuda:0 --with_std True',
    
#     ==================== svhn ====================
#     '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --device cuda:0',
#     '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --drn_k 6 --device cuda:0',
#     '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 75 --lr 0.01 --drn_k 8 --device cuda:0',
    
#     '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --device cuda:1',
#     '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --drk_k 6 --device cuda:1',
#     '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 75 --lr 0.01 --drk_k 8 --device cuda:1',
    
#     '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 200 --lr 0.01 --device cuda:1',
#     '--dataset_name svhn --hidden_dim 64 --max_epoch 100 --num_sp 200 --lr 0.01 --drn_k 6 --device cuda:1',
    
#     '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 200 --lr 0.01 --device cuda:0',
#     '--dataset_name svhn --hidden_dim 256 --max_epoch 100 --num_sp 200 --lr 0.01 --drn_k 6 --device cuda:0',
]

run_cmd = 'cd ~/SPGIE_thesis && python train.py'

for args in args_list:
    cmd = f'{run_cmd} {args}'
    os.system(cmd)

