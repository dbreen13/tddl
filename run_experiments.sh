#!/bin/bash

# CP & Tuck RN18 cifar10
for i in {1..5}; do 
    for LAYER in 15 19 28 38 41 44 60 63; do 
        for RANK in 1 25 5 75 9; do 
            for FACT in cp tucker; do 
                echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && 
                CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml";
                python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK;
            done;
        done;
    done;
done;

# TT rn18 cifar10
for i in {1..5}; do 
    for LAYER in 15 19 28 38 41 44 60 63; do 
        for RANK in 1 25 5 75 9; do 
            echo "{$i}-{$LAYER}-tt-{$RANK}" && 
            python train.py main --config-path /home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data_workers=4;  
        done; 
    done; 
done;

# CP & Tucker Garipov cifar10
for i in {1..5}; do 
    for LAYER in 2 4 6 8 10; do 
        for RANK in 1 25 5 75 9; do 
            for FACT in cp tucker; do
                echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && 
                CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml";
                python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK --factorization=$FACT; 
            done; 
        done; 
    done; 
done;

# TT Garipov Cifar
for i in {1..5}; do 
    for LAYER in 2 4 6 8 10; do 
        for RANK in 1 25 5 75 9; do 
            echo "{$i}-{$LAYER}-{$RANK}" && 
            python train.py main --config-path /home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --rank=0.$RANK --data_workers=4; 
        done; 
    done; 
done;

# CP & Tuckker Garipov Fminst
for i in {1..5}; do 
    for LAYER in 2 4 6 8 10; do 
        for RANK in 1 25 5 75 9; do 
            for FACT in cp tucker; do 
                echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && 
                CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/decompose/dec-$FACT-r0.5-$LAYER.yml";
                python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK --factorization=$FACT; 
            done; 
        done; 
    done; 
done;

# TT Garipov Fminst
for i in {1..5}; do 
    for LAYER in 2 4 6 8 10; do 
        for RANK in 1 25 5 75 9; do 
            echo "{$i}-{$LAYER}-{$RANK}" && 
            python train.py main --config-path /home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/decompose/dec-tt-r0.$RANK-$LAYER.yml --data_workers=4 --rank=0.$RANK; 
        done; 
    done; 
done;