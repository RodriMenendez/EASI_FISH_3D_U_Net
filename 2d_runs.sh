#!/bin/bash
for out_ch in 1 2
do
    for lr in 0.005 0.01 0.1
    do
    loss="bce"
        if [ ${out_ch} = 1 ]
        then
            loss="dice"
        fi
        CUDA_VISIBLE_DEVICES=1, python main.py --data_path /home/aavelarm/Data/BigNeuron/ --image_shape 128 128 --spatial_dims 2 --data_type tensor --data_len 2880 --batch_size 32 --epochs 60 --run_name 2D_unet_${loss}_lr_${lr} --model UNet --dataset BigNeuron --wandb_project BigNeuron --lr $lr --channels 32 64 128 256 512 --strides 2 2 2 2 --loss $loss --out_channels $out_ch
    done
done