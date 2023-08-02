import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import argparse

import get_data
import models
import lightning_modules

parser = argparse.ArgumentParser()
parser.add_argument('--train_idx', nargs='+', default= [0, 1, 2, 3], type=int, help='indices of files to use for training (defaults to [0, 1, 2, 3])')
parser.add_argument('--test_idx', nargs='*', default=[4, 5], type=int, help='indices of files to use for testing (defaults to [4, 5])')
parser.add_argument('--image_shape', nargs=3, default = [64, 128, 128], type=int, help='z x y shape to resize images to (defaults to [64, 128, 128])')
parser.add_argument('--data_path', type=str, required=True, help='pathfile to dataset')
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
parser.add_argument('--run_name', type=str, help='name of run')
parser.add_argument('--model', choices=['UNet3D', 'UNetR'], help='model to use (currently supports UNet3D and UNetR)')
parser.add_argument('--accelerator', choices=['auto', 'cuda', 'mps', 'cpu'], default='cpu', help='lightning accelerator (can be auto, cuda, mps, or cpu)')


args = parser.parse_args()

def main(args):
    # get the data
    easi_fish = get_data.EASI_FISH(args.train_idx, 
                                args.test_idx, 
                                args.image_shape,
                                args.data_path,
                                args.batch_size,
                                args.num_workers)

    train_data, trainloader, test_data, testloader = easi_fish

    # get model
    if args.model == 'UNet3D':
        model = models.UNet3D(args.spatial_dims, 
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
            "dataset": "EASI_FISH",
            "architecture": args.model
        }

    wandb_logger = WandbLogger(project='EASI_FISH-3DUNet-Segmentation', config=wandb_config, name=args.run_name)

    # lightning module
    autoencoder = lightning_modules.UNet3DModule(model, args.lr, args.weight_decay)

    # train
    trainer = L.pytorch.Trainer(accelerator=args.accelerator, max_epochs=args.epochs, logger=wandb_logger, log_every_n_steps=1, detect_anomaly=True)
    trainer.fit(model=autoencoder, train_dataloaders=trainloader, val_dataloaders=testloader)

    # test
    test_result = trainer.test(autoencoder, dataloaders=testloader, verbose=False)

    # save model

    path = './saved_models/' + args.run_name
    torch.save(model.state_dict(), path)



if __name__ == '__main__':
    main(args)