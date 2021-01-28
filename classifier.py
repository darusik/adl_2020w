
import argparse
import albumentations as alb
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from cnn_finetune import make_model
from lfw import LFWDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='cnn_finetune LFW')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='resnet18', metavar='M',
                    help='model name (default: resnet18)')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--train-data-dir', type=str, default='LFW',
                    help='Directory with train data')
parser.add_argument('--test-data-dir', type=str, default='LFW',
                    help='Directory with test data')
parser.add_argument('--train-data-fraq', type=float, default='0.8')
parser.add_argument('--test-data-fraq', type=float, default='0.8')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--train-image-size', type=int, default=160)
parser.add_argument('--test-image-size', type=int, default=160)

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    '''Main function to run code in this script'''

    model_name = args.model_name

    classes = (
        '0', '1'
    )

    model = make_model(
        model_name,
        pretrained=False,
        num_classes=len(classes),
        dropout_p=args.dropout_p,
        input_size=(64, 64) if model_name.startswith(('vgg', 'squeezenet')) else None,
    )
    model = model.to(device)


    transform_train = alb.Compose([
                                alb.Resize(64, 64),
                                alb.HorizontalFlip(p=0.5),
                                alb.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                                alb.RandomBrightnessContrast(p=0.5),
                                alb.GaussNoise(var_limit=(2.0, 5.0), mean=0, always_apply=False, p=0.3),
                                alb.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=False, p=0.3),
                                alb.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=50, val_shift_limit=30, always_apply=False, p=0.5),
                                alb.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
                                alb.Normalize(),
                                ToTensorV2(),
                                ])

    transform_test = alb.Compose([
                                alb.Resize(64, 64),
                                alb.Normalize(),
                                ToTensorV2(),
                                ])

    train_loader = torch.utils.data.DataLoader(
        LFWDataset(data_folder=args.train_data_dir,
                    image_size=args.train_image_size,
                    data_slice=[0, args.train_data_fraq],
                    transform=transform_train
                    ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(data_folder=args.test_data_dir,
                    image_size=args.test_image_size,
                    data_slice=[args.test_data_fraq, 1],
                    transform=transform_test
                    ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Train
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(model, epoch, optimizer, train_loader)
        test(model, test_loader)

    torch.save(model.state_dict(), "models/mask_"+str(args.model_name)+"_epochs_"+str(args.epochs)+"_adam.pth")

if __name__ == '__main__':
    main()
