import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import argparse

import get_data
import models
import lightning_modules

parser = argparse.ArgumentParser()
parser.add_argument('--data_split_ratios', nargs=3, default= [0.8, 0.1, 0.1], type=float, help='ratio of train to test to validation data (defaults to [0.8, 0.1, 0.1])')
parser.add_argument('--image_shape', nargs='+', default = [64, 128, 128], type=int, help='z x y shape to resize images to (defaults to [64, 128, 128])')
parser.add_argument('--data_path', type=str, required=True, help='pathfile to dataset (required). Assumes that ground truth images are located at data_path/images and labels are located at data_path/labels')
parser.add_argument('--data_type', type=str, choices=['tif', 'tensor'], default='tif', help='type of data (can be tif or tensor)')
parser.add_argument('--data_len', type=int, default=None, help='length of dataset (defaults to None)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training (defaults to 1)')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers for dataloader (defaults to 1)')
parser.add_argument('--lr', default=1e-3, type=float, help='learning Rate (defaults to 1e-3)')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay(defaults to 0)')
parser.add_argument('--spatial_dims', default=3, type=int, help='number of spatial dimensions of model (defaults to 3)')
parser.add_argument('--in_channels', default = 1, type=int, help='number of input channels (defaults to 1)')
parser.add_argument('--out_channels', default=1, type=int, help='number of output channels (defaults to 1)')
parser.add_argument('--channels', nargs='+', default = [4, 8, 16], type=int, help='sequence of channels. Top block first. The length of channels should be no less than 2 (defaults to [4, 8, 16])')
parser.add_argument('--strides', nargs='+', default=[2, 2], type=int, help='sequence of convolution strides. The length of stride should equal to len(channels) - 1 (defaults to [2, 2])')
parser.add_argument('--kernel_size', nargs='*', default=3, type=int, help='convolution kernel size, the value(s) should be odd. If sequence, its length should equal to dimensions (defaults to 3)')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train for (defaults to 10)')
parser.add_argument('--run_name', type=str, required=True, help='name of run (required)')
parser.add_argument('--model', choices=['UNet', 'UNetR'], help='model to use (currently supports UNet3D and UNetR)')
parser.add_argument('--accelerator', choices=['auto', 'cuda', 'mps', 'cpu'], default='cpu', help='lightning accelerator (can be auto, cuda, mps, or cpu)')
parser.add_argument('--seed', type=int, default=0, help='seed for random data splitting (defaults to 0)')
parser.add_argument('--dataset', type=str, default=None, help='name of dataset for wandb (defaults to None)')
parser.add_argument('--wandb_project', type=str, required=True, help='name of wandb project (required)')
parser.add_argument('--loss', choices=['dice', 'bce'], default='dice', help='loss function to use (currently supports dice loss and cross entropy loss)')

args = parser.parse_args()

def main(args):
    # get the data
    dataset = get_data.CreateDataset(args.data_split_ratios,
                                     args.image_shape,
                                     args.data_path,
                                     args.data_type,
                                     args.data_len,
                                     args.batch_size,
                                     args.num_workers,
                                     args.seed,
                                     args.spatial_dims)

    train_data, trainloader, test_data, testloader, val_data, valloader = dataset

    # get model
    if args.model == 'UNet':
        model = models.U_Net(args.spatial_dims, 
                                args.in_channels, 
                                args.out_channels, 
                                args.channels, 
                                args.strides, 
                                args.kernel_size)
    elif args.model == 'UNetR':
        model = models.UNetR(args.in_channels,
                             args.out_channels,
                             args.image_shape)

    # wandb
    wandb_config = {
            "dataset": args.dataset,
            "architecture": args.model
        }

    wandb_logger = WandbLogger(project=args.wandb_project, config=wandb_config, name=args.run_name)

    # lightning module
    autoencoder = lightning_modules.UNet3DModule(model, args.lr, args.weight_decay, loss = args.loss)

    # train
    trainer = L.pytorch.Trainer(accelerator=args.accelerator, max_epochs=args.epochs, logger=wandb_logger, log_every_n_steps=1, detect_anomaly=True)
    trainer.fit(model=autoencoder, 
                train_dataloaders=trainloader if train_data.exists else None, 
                val_dataloaders=valloader if val_data.exists else None)

    # test
    test_result = trainer.test(autoencoder, 
                               dataloaders=testloader if test_data.exists else None, 
                               verbose=False)

    # save model

    path = './saved_models/' + args.run_name
    torch.save(model.state_dict(), path)



if __name__ == '__main__':
    main(args)