import torch
from torch import nn

def SeqConvNet(D_out,fixed_dropout=None,in_channels=1,channels=(6,16),h_dims=(120,84)):
    print("Using SeqConvNet")
    assert len(channels) == 2 == len(h_dims)
    pool = lambda: nn.MaxPool2d(2,2)
    dropout = lambda: torch.nn.Dropout(p=fixed_dropout)
    dropout_li = lambda: ([] if fixed_dropout is None else [dropout()])
    relu = lambda: torch.nn.ReLU(inplace=False)
    convs = [nn.Conv2d(in_channels, channels[0], 5),nn.Conv2d(channels[0], channels[1], 5)]
    fcs = [nn.Linear(channels[1] * 4 * 4, h_dims[0]),nn.Linear(h_dims[0], h_dims[1])]
    final_fc = nn.Linear(h_dims[1], D_out)
    conv_blocks = [[conv,relu(),pool()] + dropout_li() for conv in convs]
    fc_blocks = [[fc,relu()] + dropout_li() for fc in fcs]
    flatten = lambda l: [item for sublist in l for item in sublist]
    return torch.nn.Sequential(
        *flatten(conv_blocks),
        torch.nn.Flatten(),
        *flatten(fc_blocks),
        final_fc
    )
