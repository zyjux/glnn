import torch
from torch import nn
from torchvision.transforms.functional import rotate


class glp_rotation_stack(nn.Module):
    def __init__(self, num_angles=None, angle_inc=None):
        super().__init__()
        if angle_inc is not None:
            self.num_angles = 360 / angle_inc
            self.angle_inc = angle_inc
        elif num_angles is not None:
            self.angle_inc = 360 / num_angles
            self.num_angles = num_angles
        else:
            raise ValueError("Need either num_angles or angle_inc to be defined")

    def forward(self, image):
        temp_list = [
            rotate(image, self.angle_inc * i) for i in range(int(self.num_angles))
        ]
        return torch.stack(temp_list, dim=-1)


class glp_rotation_pool(nn.Module):
    def __init__(self, glp_pooling_size: int, **kwargs):
        super().__init__()
        self.kernel_size = glp_pooling_size

    def forward(self, image):
        num_angles = image.shape[-1]
        angle_increment = 360 / num_angles
        output_image = torch.empty(image.shape, device=image.device)
        for i in range(num_angles):
            relative_offset = i % self.kernel_size
            output_image[..., i] = rotate(
                image[..., i], -angle_increment * relative_offset
            )
        return nn.MaxPool3d(kernel_size=(1, 1, self.kernel_size))(output_image)


class simple_model(nn.Module):
    def __init__(
        self,
        G_angle_inc: int,
        num_lambda_cosets: int,
        psi_size: tuple[int, int],
        do_pool: bool = True,
    ):
        super().__init__()
        # padding = int(psi_size / 2)

        self.setup_layer = glp_rotation_stack(angle_inc=G_angle_inc)
        self.convolution = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(*psi_size, 1),
            padding=0,
            # padding_mode="reflect",
        )
        self.pooling = glp_rotation_pool(num_lambda_cosets)
        self.do_pool = do_pool

    def forward(self, x):
        setup_out = self.setup_layer(x)
        conv_out = self.convolution(setup_out)
        if self.do_pool:
            pool_out = self.pooling(conv_out)
            return pool_out
        else:
            return conv_out
