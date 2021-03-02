import numpy as np
import torch
import torch.nn as nn


class YOLOV0Loss(nn.Module):
    def __init__(self, device, lambda_coor=5.0, lambda_noobj=0.5):
        super(YOLOV0Loss, self).__init__()
        self.device = device
        self.lambda_coor = torch.tensor(lambda_coor, dtype=torch.float32)
        self.lambda_noobj = torch.tensor(lambda_noobj, dtype=torch.float32)

    def forward(self, outputs, targets):
        assert outputs.shape == targets.shape, \
            f"outputs shape[{outputs.shape}] not equal targets shape[{targets.shape}]"

        b, w, h, c = outputs.shape
        # loss = torch.tensor(0.0, dtype=torch.float32)
        loss = 0.
        # loss.to(self.device)

        for bi in range(b):
            for wi in range(w):
                for hi in range(h):
                    detect_vector = outputs[bi][wi][hi]
                    gt_dv = targets[bi][wi][hi]
                    conf_pred, x_pred, y_pred, w_pred, h_pred = detect_vector
                    conf_gt, x_gt, y_gt, w_gt, h_gt = gt_dv

                    if conf_gt > 0:
                        loss = loss + self.lambda_coor * \
                            (torch.pow(x_pred-x_gt, 2) + torch.pow(y_pred-y_gt, 2))
                        loss = loss + self.lambda_coor * \
                            (torch.pow(w_pred-w_gt, 2) + torch.pow(h_pred-h_gt, 2))
                        # (torch.pow(torch.sqrt(w_pred)-torch.sqrt(w_gt), 2) +
                        #  torch.pow(torch.sqrt(h_pred)-torch.sqrt(h_gt), 2))
                        loss = loss + torch.pow(conf_pred-torch.tensor(1), 2)
                    else:
                        loss = loss + self.lambda_noobj * \
                            torch.pow(conf_pred-torch.tensor(0), 2)
        return loss/outputs.size(0)


def main():
    np.random.seed(110)
    outputs_array = np.random.rand(8, 8, 8, 5)
    targets_array = np.random.rand(8, 8, 8, 5)
    outputs = torch.tensor(outputs_array, dtype=torch.float32)
    targets = torch.tensor(targets_array, dtype=torch.float32)
    criterion = YOLOV0Loss(torch.device('cuda:0'))
    loss = criterion(
        outputs, targets)
    print(loss)


if __name__ == '__main__':
    main()
