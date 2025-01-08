import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from data.noisedata import NoiseData, NoiseDataFFT
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin, NonLinearTypeBinModel
from utils.transform import Normalizer
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=500, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.01, type=float)
    parser.add_argument('--lr_decay', type = list, default = [100,200,300,400], help = 'learning rate decay')
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_fft_train.xlsx', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--log_dir', dest='log_dir', type = str, default = 'logs/train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    if args.dataset == 'NoiseData':
        dataset = NoiseDataFFT(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=True, fft_out=80)

    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    

    model = NonLinearTypeBinModel(nc=400, out_nc=14, num_bins=80, num_sheets=4)
    if args.snapshot != '':
        saved_state_dict = torch.load(args.snapshot, weights_only=True)
        model.load_state_dict({name: weight for name, weight in saved_state_dict.items() if name.startswith('hidden')}, strict=False)
    
    criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss(reduction='sum')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    milestones = args.lr_decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # tensorboard visualization
    Loss_writer = SummaryWriter(log_dir = args.log_dir)

    for epoch in range(args.num_epochs):
        for i, (inputs, outputs, types, sheet_idx) in tqdm(enumerate(train_loader)):
            inputs = Variable(inputs)
            labels = Variable(outputs)
            optimizer.zero_grad()
            preds = model(inputs)
            batch_indices = torch.arange(preds.size(0))
            preds = preds[batch_indices, sheet_idx.squeeze(), :, :]
            types = types.view(-1, 1, 1)
            preds = preds.gather(1, types.expand(-1, 1, preds.size(2)))
            preds = preds.squeeze(1)

            # calculate loss
            loss_flag = torch.ones(inputs.size(0))
            cos_loss = cos_criterion(preds, labels, loss_flag)
            mse_loss = criterion(preds, labels)
            loss = cos_loss + mse_loss
            loss.backward()
            optimizer.step()

            Loss_writer.add_scalar('train_loss', loss, epoch)
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f cos_loss: %.4f mse_loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss, cos_loss, mse_loss))
            # Save models at numbered epochs.

        scheduler.step()
        if epoch % 100 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            if not os.path.exists('snapshots/'):
                os.makedirs('snapshots/')
            torch.save(model.state_dict(),
            'snapshots/' + args.output_string + '_epoch_'+ str(epoch) + '.pth')