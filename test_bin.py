import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData, NoiseDataBin
from utils.transform import Normalizer
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin, NonLinearMultiBin
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_test.xlsx', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    dataset = NoiseDataBin(dir=args.data_dir, filename=args.filename, transform=transformations, use_type=True)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearTypeBin(nc=800, out_nc=18, num_bins=51)
    saved_state_dict = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(saved_state_dict)

    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    idx_tensor = [idx for idx in range(51)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor))

    criterion = nn.MSELoss()
    test_error = .0
    total = 0
    for i, (inputs, outputs, bins, bins0, bins1, bins2, types) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = Variable(inputs)
        labels = Variable(outputs)
        bin_labels = Variable(bins).squeeze()
        # bin0_labels = Variable(bins0).squeeze()
        # bin1_labels = Variable(bins1).squeeze()
        # bin2_labels = Variable(bins2).squeeze()
        preds = model(inputs, types)

        # Binned predicitions
        _, bpred = torch.max(preds, 1)

        # regression predictions
        preds_reg = F.softmax(preds, dim=1)
        preds_reg = torch.sum(preds_reg * idx_tensor, 1) * 1 + 20 - 0.5
        preds_reg = preds_reg.unsqueeze(1)

        # print(bpred, bins, preds_reg, labels)
        test_loss = criterion(preds_reg, labels)
        test_error += torch.sum(test_loss)
    
    print('Test error on the ' + str(total) +' test samples. MSE: %.4f' % (test_error * batch_size/ total))