import os
import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision


class trainvalDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, labels_dir, subset_path, input_size, grid_num, pretrained):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.grid_num = grid_num
        self.img_files = []
        self.label_files = []
        self.pretrained = pretrained

        with open(subset_path) as f:
            for line in f:
                self.img_files.append(line.strip())
                self.label_files.append(line.strip().replace('.jpg', '.txt'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.img_files[idx])
        img = Image.open(img_path)
        img = img.resize((self.input_size, self.input_size))
        img = np.array(img).transpose(2, 1, 0)
        if self.pretrained:
            img = torch.tensor(img, dtype=torch.float32)
            transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
            img = transform(img)
        else:
            img = img/255
            img = torch.tensor(img, dtype=torch.float32)

        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        boxes = []
        with open(label_path) as f:
            for line in f:
                _, cx, cy, w, h = list(map(float, line.strip().split()))
                boxes.append((cx, cy, w, h))

        target = torch.zeros(self.grid_num, self.grid_num,
                             5, dtype=torch.float32)
        grid_size = self.input_size//self.grid_num
        for box in boxes:
            cx, cy, w, h = box
            cell_x = int(((cx)*self.input_size)//grid_size)
            cell_y = int(((cy)*self.input_size)//grid_size)
            obj_vector = target[cell_x][cell_y]

            obj_vector[0] = 1.
            obj_vector[1] = (((cx)*self.input_size) % grid_size)/grid_size
            obj_vector[2] = (((cy)*self.input_size) % grid_size)/grid_size
            obj_vector[3] = w
            obj_vector[4] = h

        return img, target

    def show_info(self):
        print(f'imgs dir: {self.imgs_dir}')
        print(f'img files: {self.img_files[:4]} ...')
        print(f'labels dir: {self.labels_dir}')
        print(f'label files: {self.label_files[:4]}')
        print(f'files count: {len(self.img_files)}')
        print(f'image shape: {self.input_size} * {self.input_size}')
        print(f'target shape: {self.grid_num} * {self.grid_num} * {5}')


def main():
    imgs_dir = '/data_nas/yckj3341/dataset/hlw/JPEGImages'
    labels_dir = '/data_nas/yckj3341/dataset/hlw/labels'
    subset_path = '/data_nas/yckj3341/dataset/hlw/ImageSet/train.txt'
    img_size = 512
    grid_num = 8
    pretrained = False
    dataset = trainvalDataset(imgs_dir,
                              labels_dir,
                              subset_path,
                              img_size,
                              grid_num,
                              pretrained)
    dataset.show_info()
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1)
    for iteration, (input_tensor, target_tensor) in enumerate(train_loader):
        img = input_tensor[0]
        img = img.numpy()
        img = img*255
        img = img.transpose(2, 1, 0)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(img)

        grid_size = img_size/grid_num
        for batch_i in range(1):
            for col in range(grid_num):
                for row in range(grid_num):
                    vec = target_tensor[batch_i][col][row]
                    if vec[0] > 0:
                        _, cx_, cy_, w_, h_ = vec
                        cx = (col+cx_)*grid_size
                        cy = (row+cy_)*grid_size
                        w = w_*img_size
                        h = h_*img_size
                        draw.rectangle([int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)],
                                       outline=(0, 255, 0), width=2)

        img.save('dataset_test.jpg')
        break


if __name__ == '__main__':
    main()
