#!/bin/bash
for transformation in 'nearest'
do
    CUDA_VISIBLE_DEVICES=1, python main.py --masks_path /data/AllCellData/neuron/Vaa3D_masks/ --inputs_path /data/AllCellData/neuron/gold_166_images/ --image_shape 128 256 256 --spatial_dims 3 --batch_size 1 --epochs 100 --run_name 3D_unet_method_${transformation}_bigger_model --model UNet --dataset BigNeuron --wandb_project BigNeuron --lr 0.005 --channels 64 128 256 512 --strides 2 2 2 --loss dice --patience 15 --out_channels 1 --transforms ${transformation} --custom_dataset vaa3d --data_len 89 --accelerator cuda
done
