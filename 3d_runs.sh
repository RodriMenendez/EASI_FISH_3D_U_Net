#!/bin/bash
for out_ch in 1
do
    for lr in 0.005 0.01
    do
    loss="bce"
        if [ ${out_ch} = 1 ]
        then
            loss="dice"
        fi
        for batch_size in 1 2
        do
            CUDA_VISIBLE_DEVICES=1, python main.py --data_path /home/aavelarm/Data/BigNeuron/ --image_shape 32 128 128 --spatial_dims 3 --batch_size ${batch_size} --epochs 100 --run_name 3D_unet_${loss}_lr_${lr}_batch_${batch_size} --model UNet --dataset BigNeuron --wandb_project BigNeuron --lr $lr --channels 32 128 256 --strides 2 2 --loss $loss --patience 20 --out_channels $out_ch --custom_dataset vaa3d_custom_data
        done
    done
done