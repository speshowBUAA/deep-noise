import argparse
from data.noisedata import NoiseData, NoiseDataFFT
from utils.transform import Normalizer
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sub
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_fft_train.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    args = parser.parse_args()
    return args

def stairs(x,y):
    X = []
    Y = []
    X.append(x[0])
    X.append(x[0])
    Y.append(y[0])
    Y.append(y[0])
    for i in range(1, len(x)):
        X.append(x[i])
        X.append(x[i])
        Y.append(y[i-1])
        Y.append(y[i])
    return X, Y

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    if args.dataset == 'NoiseData':
        dataset = NoiseDataFFT(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=True, fft_out=80)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearTypeBin(nc=400, out_nc=14, num_bins=80)
    saved_state_dict = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(saved_state_dict)
    model.eval()
    
    criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss(reduction='sum')

    input, label, type_  = dataset.__getitem__(np.random.randint(dataset.__len__()))
    input = input.unsqueeze(0)
    type_ = type_.unsqueeze(0)
    label = label.unsqueeze(0)
    pred = model(input, type_)
    # calculate loss
    loss_flag = torch.ones(input.size(0))
    cos_loss = cos_criterion(pred, label, loss_flag)
    mse_loss = criterion(pred, label)

    pred = pred.detach().squeeze()
    label = label.squeeze()
    y = np.array(range(0,5000,64))

    fig = go.Figure()
    s_x, s_y = stairs(y, label)
    fig.add_trace(go.Scatter(x=s_x, y=s_y, mode='lines', name='label'))
    s_x, s_y = stairs(y, pred)
    fig.add_trace(go.Scatter(x=s_x, y=s_y, mode='lines', name='pred'))

    # fig = sub.make_subplots(rows=2, cols=1, subplot_titles=['dB', 'Pa'])
    # fig.add_trace(go.Scatter(x=y, y=label, mode='lines', name='label'), 1, 1)
    # fig.add_trace(go.Scatter(x=y, y=pred, mode='lines', name='pred'), 1, 1)
    # s_x, s_y = stairs(y, 10**label)
    # fig.add_trace(go.Scatter(x=s_x, y=s_y, mode='lines', name='label'), 2, 1)
    # s_x, s_y = stairs(y, 10**pred)
    # fig.add_trace(go.Scatter(x=s_x, y=s_y, mode='lines', name='pred'), 2, 1)
    # fig.update_yaxes(tickformat=".2f", row=1)
    # fig.update_yaxes(tickformat=".2e", row=2)

    fig.update_yaxes(tickformat=".2f")
    fig.update_xaxes(tickformat="%d")
    fig.update_layout(plot_bgcolor="#fff")
    fig.write_html('fft_demo_cos_%.4f_mse_%.4f.html'%(cos_loss, mse_loss), full_html=False, include_plotlyjs='cdn')
