import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision

from models import YOLOV0


def predict(model, img_path, device, output_path):
    img_size = 512
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img = np.array(img).transpose(2, 1, 0)
    img = torch.tensor(img, dtype=torch.float32)
    transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    x = transform(img)
    # x = torch.tensor(img, dtype=torch.float32, device=device)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    pre_tensor = model(x)
    box_l = []
    B, Cell_W, Cell_H, Cell_C = pre_tensor.shape
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    draw = ImageDraw.Draw(img)
    for batch_i in range(B):
        for col in range(Cell_W):
            for row in range(Cell_H):
                cell_vec = pre_tensor[batch_i][col][row]
                box_l.append((cell_vec, col, row))

    box_l.sort(key=lambda x: x[0][0], reverse=True)
    grid_size = img_size//8
    print(box_l[0])
    for box in box_l[:1]:
        (conf, x_, y_, w_, h_), col, row = box
        x = (col+x_)*grid_size
        y = (row+y_)*grid_size
        w = w_*img_size
        h = h_*img_size
        draw.rectangle([int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)],
                       outline=(255, 0, 0), width=3)
        img.save(output_path)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = YOLOV0('vgg16', False)
    model.load_state_dict(torch.load(
        'save_weights/yolov0_vgg16_pretrained_14.pth'))
    model.to(device)
    model.eval()
    files = []
    with open('val.txt') as f:
        for line in f:
            files.append(line.strip())
    for i, file in enumerate(files):
        print(i)
        predict(model,
                f'JPEGImages/{file}',
                device,
                f'test_res/test_res_{file}')
