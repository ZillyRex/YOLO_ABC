import os
import argparse
import matplotlib.pyplot as plt
import time

import torch
from torch.optim import SGD

from dataset import trainvalDataset
from models import YOLOV0
from loss import YOLOV0Loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(backbone, pretrained, img_dir, label_dir, subset_path, input_size, grid_num, batch_size, loss_dir):
    if pretrained == 'True':
        pretrained_str = 'pretrained'
        pretrained = True
    else:
        pretrained_str = 'random'
        pretrained = False

    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    epochs = 15

    dataset = trainvalDataset(
        img_dir, label_dir, subset_path, int(input_size), int(grid_num), pretrained)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True, num_workers=8)

    yolov0 = YOLOV0(backbone, pretrained)
    yolov0.train()
    yolov0.to(device)

    criterion = YOLOV0Loss(device)
    optimizer = SGD(yolov0.parameters(), lr=0.01, momentum=0.9)

    loss_l = []
    for epoch in range(epochs):
        ts = time.time()
        if epoch == 2:
            optimizer = SGD(yolov0.parameters(), lr=0.001, momentum=0.9)
        if epoch == 6:
            optimizer = SGD(yolov0.parameters(), lr=0.0001, momentum=0.9)
        for iteration, (input_tensor, target_tensor) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = yolov0(input_tensor.to(device))
            loss = criterion(
                outputs.to(device), target_tensor.to(device))
            loss.backward()
            optimizer.step()

            loss_l.append(loss.item())
            plt.plot(loss_l)
            plt.savefig(os.path.join(
                loss_dir, f'loss_{backbone}_{pretrained_str}_{input_size}_{grid_num}_{batch_size}.jpg'))
            plt.close()

            if iteration % 10 == 0:
                print(
                    f"epoch{epoch}, iter{iteration}, loss: {loss.item():.4f}, lr: {get_lr(optimizer)}")

        # torch.save(yolov0.state_dict(),
        #            f'backup/yolov0_{backbone}_{pretrained_str}_{input_size}_{grid_num}_{batch_size}_{epoch}.pth')
        print(f"Finish epoch {epoch}, time elapsed {(time.time() - ts):.2f}s")
        print("*"*30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--label_dir', type=str)
    parser.add_argument('--subset_path', type=str)
    parser.add_argument('--input_size', type=str)
    parser.add_argument('--grid_num', type=str)
    parser.add_argument('--batch_size', type=str)
    parser.add_argument('--loss_dir', type=str)

    args = parser.parse_args()
    train(args.backbone,
          args.pretrained,
          args.img_dir,
          args.label_dir,
          args.subset_path,
          args.input_size,
          args.grid_num,
          args.batch_size,
          args.loss_dir)


if __name__ == '__main__':
    main()
