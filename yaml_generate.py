import os

template = """\
# data
dataset: cifar10

# model
model_name: rn18
baseline_path: /home/demi/Documents/tddl/pretrained/cifar10/logs/rn18/baselines/1646417743/rn18_18_dNone_128_sgd_l0.1_g0.1_sTrue/cnn_best.pth

# training
batch: 128
epochs: 25
optimizer: 'adam'
lr: 1.0e-5
gamma: 0

# factorization
factorization: '{factorization}'
decompose_weights: true
rank: {rank}
layers:
- {layer}

# datalogging
data_dir: "/home/demi/Documents/tddl/bigdata/cifar10" 
logdir: "/home/demi/Documents/tddl/bigdata/cifar10/logs/rn18/decomposed" 

# hardware
data_workers: /home/demi/Documents/tddl/pretrained/cifar10/logs/rn18/baselines/1646417743/rn18_18_dNone_128_sgd_l0.1_g0.1_sTrue
"""

factorizations = ['cp', 'tucker']
ranks = [0.1, 0.25, 0.75, 0.9]
layers = [15, 19, 28, 38, 41, 44, 60, 63]

output_dir = "/home/demi/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_count = 0
for factorization in factorizations:
    for rank in ranks:
        for layer in layers:
            file_name = f"dec-{factorization}-r{rank}-{layer}.yml"
            file_content = template.format(factorization=factorization, rank=rank, layer=layer)
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w") as f:
                f.write(file_content)
            file_count += 1

print(f"{file_count} files generated.")


# import os

# template = """\
# # data
# dataset: cifar10

# # model
# model_name: gar
# baseline_path: /home/demi/Documents/tddl/pretrained/cifar10/logs/garipov/baselines/1647358615/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue/cnn_best.pth

# # training
# batch: 128
# epochs: 10
# optimizer: 'sgd'
# momentum: 0.9
# lr: 1.0e-4
# gamma: 0
# weight_decay: 0

# # factorization
# factorization: '{factorization}'
# decompose_weights: true
# rank: {rank}
# layers:
# - {layer}

# # datalogging
# data_dir: "/home/demi/Documents/tddl/bigdata/cifar10" 
# logdir: "/home/demi/Documents/tddl/bigdata/cifar10/logs/garipov/decomposed" 

# # hardware
# data_workers: 8,
# """

# factorizations = ['cp', 'tucker']
# ranks = {'cp': [0.1, 0.25, 0.75, 0.9], 'tucker': [0.1, 0.25, 0.5, 0.75, 0.9]}
# layers = [2, 4, 6, 8, 10]

# output_dir = "/home/demi/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose" 

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# file_count = 0
# for factorization in factorizations:
#     for rank in ranks[factorization]:
#         for layer in layers:
#             file_name = f"dec-{factorization}-r{rank}-{layer}.yml"
#             file_content = template.format(factorization=factorization, rank=rank, layer=layer)
#             file_path = os.path.join(output_dir, file_name)
#             with open(file_path, "w") as f:
#                 f.write(file_content)
#             file_count += 1

# print(f"{file_count} files generated.")


# import os

# template = """\
# # data
# dataset: fmnist

# # model
# model_name: gar
# baseline_path: /home/demi/Documents/tddl/pretrained/f_mnist/logs/garipov/baselines/1647955843/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue/cnn_best.pth

# # training
# batch: 128
# epochs: 10
# optimizer: 'sgd'
# momentum: 0.9
# lr: 1.0e-4
# gamma: 0
# weight_decay: 0

# # factorization
# factorization: '{factorization}'
# decompose_weights: true
# rank: {rank}
# layers:
# - {layer}

# # datalogging
# data_dir: "/home/demi/Documents/tddl/bigdata/f_mnist" 
# logdir: "/home/demi/Documents/tddl/bigdata/f_mnist/logs/garipov/decomposed" 

# # hardware
# data_workers: 8
# """

# factorizations = ['cp', 'tucker']
# ranks = {'cp': [0.1, 0.25, 0.75, 0.9], 'tucker': [0.1, 0.25, 0.5, 0.75, 0.9]}
# layers = [2, 4, 6, 8, 10]

# output_dir = "/home/demi/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/decompose"

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# file_count = 0
# for factorization in factorizations:
#     for rank in ranks[factorization]:
#         for layer in layers:
#             file_name = f"dec-{factorization}-r{rank}-{layer}.yml"
#             file_content = template.format(factorization=factorization, rank=rank, layer=layer)
#             file_path = os.path.join(output_dir, file_name)
#             with open(file_path, "w") as f:
#                 f.write(file_content)
#             file_count += 1

# print(f"{file_count} files generated.")
