# imports

import torch
import torch.nn as nn


class CustomCNN(nn.Module):

    def __init__(self, cfg):
        super(CustomCNN, self).__init__()

        # Extract configuration parameters
        self.num_classes = cfg.num_classes
        self.input_channels = cfg.input_channels
        self.use_batch_norm = cfg.use_batch_norm
        self.use_dropout = cfg.use_dropout
        self.dropout_rate = cfg.dropout_rate

        num_layers = cfg.num_layers
        kernel_sizes = cfg.kernel_size
        strides = cfg.stride
        fc_layers_config = cfg.fc_layers

        # Validate that all lists have the same length
        n_conv_layers = len(num_layers)
        if len(kernel_sizes) != n_conv_layers or len(strides) != n_conv_layers:
            raise ValueError(f"num_layers, kernel_size, and stride must have the same length. "
                           f"Got {n_conv_layers}, {len(kernel_sizes)}, {len(strides)}")

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_channels

        for i in range(n_conv_layers):
            out_channels = num_layers[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]

            block = []
            block.append(nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding='same'))

            if self.use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))

            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if self.use_dropout:
                block.append(nn.Dropout2d(p=self.dropout_rate))

            self.conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()

        # First FC layer takes flattened conv output
        if fc_layers_config:
            fc_in_features = in_channels

            for i, fc_size in enumerate(fc_layers_config):
                fc_block = []
                fc_block.append(nn.Linear(fc_in_features, fc_size))
                fc_block.append(nn.ReLU(inplace=True))

                if self.use_dropout:
                    fc_block.append(nn.Dropout(p=self.dropout_rate))

                self.fc_layers.append(nn.Sequential(*fc_block))
                fc_in_features = fc_size

            self.classifier = nn.Linear(fc_in_features, self.num_classes)
        else:
            self.classifier = nn.Linear(in_channels, self.num_classes)

    def forward(self, x):

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Classification
        x = self.classifier(x)

        return x

    def get_num_parameters(self):
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cnn_from_config(config):
    return CustomCNN(config)