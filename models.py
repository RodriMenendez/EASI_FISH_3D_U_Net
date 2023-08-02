from monai.networks.nets import UNet, UNETR

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

def UNetR(in_channels, out_channels, img_size):
    model = UNETR(in_channels, out_channels, img_size)

    return model