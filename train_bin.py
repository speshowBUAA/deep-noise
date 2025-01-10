import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch import nn, optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from data.noisedata import NoiseData, NoiseDataBin
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin, NonLinearMultiBin
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
          default='data_final_train.xlsx', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--log_dir', dest='log_dir', type = str, default = 'logs/train')
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.1, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    dataset = NoiseDataBin(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=True)

    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    model = NonLinearTypeBin(nc=800, out_nc=18, num_bins=51)
    criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    milestones = args.lr_decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Regression loss coefficient
    alpha = args.alpha

    # tensorboard visualization
    Loss_writer = SummaryWriter(log_dir = args.log_dir)

    idx_tensor = [idx for idx in range(51)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor))

    for epoch in range(args.num_epochs):
        for i, (inputs, outputs, bins, bins0, bins1, bins2, types) in tqdm(enumerate(train_loader)):
            inputs = Variable(inputs)
            labels = Variable(outputs)
            bin_labels = Variable(bins).squeeze()
            # bin0_labels = Variable(bins0).squeeze()
            # bin1_labels = Variable(bins1).squeeze()
            # bin2_labels = Variable(bins2).squeeze()
            optimizer.zero_grad()
            preds = model(inputs, types)

            # calculate Cross entropy loss
            loss = criterion(preds, bin_labels)
            # loss0 = criterion(preds0, bin0_labels)
            # loss1 = criterion(preds1, bin1_labels)
            # loss2 = criterion(preds2, bin2_labels)
            loss.backward()
            optimizer.step()

            # calculate MSE loss
            preds_cont = F.softmax(preds, dim=1)
            preds_cont = torch.sum(preds_cont * idx_tensor, 1) * 1 + 20 - 0.5
            preds_cont = preds_cont.unsqueeze(1)
            loss_reg = reg_criterion(preds_cont, outputs)

            # Total loss
            total_loss = alpha * loss_reg + loss

            Loss_writer.add_scalar('train_loss', total_loss, epoch)
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] total_loss: %.4f Losses_reg: %.4f Losses: %.4f'
                       %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, total_loss, loss_reg, loss))
            # Save models at numbered epochs.

        scheduler.step()
        if epoch % 100 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            if not os.path.exists('snapshots/'):
                os.makedirs('snapshots/')
            torch.save(model.state_dict(),
            'snapshots/' + args.output_string + '_epoch_'+ str(epoch) + '.pth')