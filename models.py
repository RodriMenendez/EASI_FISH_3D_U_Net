from monai.networks.nets import UNet

def UNet3D(spatial_dims, in_channels, out_channels, channels, strides, kernel_size):
    model = UNet(
    spatial_dims,
    in_channels,
    out_channels,
    channels,
    strides,
    kernel_size
    )

    return model