import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData
from utils.transform import Normalizer
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeModel
import torch
from torch.autograd import Variable
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_test.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--nc', dest='nc', type = int, default = 3200)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

    if args.dataset == 'NoiseData':
        dataset = NoiseData(dir=args.data_dir, filename='data_final_test.xlsx', transform=transformations, use_type=True)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearTypeModel(nc=args.nc).to(device)
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)
    
    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    criterion = nn.MSELoss().to(device)
    test_error = .0
    total = 0

    for i, (inputs, outputs, types, sheet_idx) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = inputs.to(device)
        labels = outputs.to(device)
        types = types.long().to(device)
        sheet_idx = sheet_idx.to(device)
        
        preds = model(inputs)
        
        batch_indices = torch.arange(preds.size(0), device=device)
        preds = preds[batch_indices, sheet_idx.squeeze(), :]
        types = types.view(-1, 1)
        preds = preds.gather(1, types)
       
        test_loss = criterion(preds, labels)
        test_error += torch.sum(test_loss)
        # print(preds, labels, test_loss, torch.sum(test_loss))
    
    print('Test error on the ' + str(total) +' test samples. MSE: %.4f' % (test_error * batch_size/ total))