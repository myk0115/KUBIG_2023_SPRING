import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.util import load_state_dict
from dataset import KeypointsDataset
from dataset import vis_heatmaps
import configs.ViTPose_huge_coco_256x192 as cfg
from models.model import ViTPose
from models.losses import JointsMSELoss


def main():
    transform = A.Compose([A.CenterCrop(864, 1152),
                           A.Resize(256, 192),
                           A.ShiftScaleRotate(rotate_limit=30, p=0.3),
                           # A.VerticalFlip(0.5),
                           A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ToTensorV2()
                           ],
                          keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'],
                                                           remove_invisible=True, angle_in_degrees=True))
    datasets = KeypointsDataset(root_dir='../dataset/',
                                target_csv='train_df.csv',
                                transforms=transform,
                                input_size=[256, 192],
                                output_size=[256, 192],
                                mode='train'
                                )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = ViTPose(cfg=cfg.model)
    model.keypoint_head.final_layer = nn.Conv2d(256, 24, kernel_size=(1, 1), stride=(1, 1))
    a = torch.load('vitpose-h-multi-coco.pth')
    model.on_load_checkpoint(a['state_dict'])  # changed torch.nn.module
    model.to(device)
    criterion = JointsMSELoss(use_target_weight=False).cuda()
    optimizer = AdamW(model.parameters(), betas=cfg.optimizer['betas'], weight_decay=cfg.optimizer['weight_decay'])
    train_loader = DataLoader(datasets, batch_size=1)

    for param in list(model.named_parameters()):
        if param[0] == 'keypoint_head.final_layer.weight' or param[0] == 'keypoint_head.final_layer.bias':
            continue

        elif param[0] == 'backbone.pos_embed':
            continue

        elif 'keypoint_head' in str(param[0]):
            continue

        else:
            param[1][1].requires_grad = False

    for epoch in range(30):
        for i, (input, target, joints) in enumerate(train_loader):
            #print(input.shape, target.shape, joints.shape)
            input = torch.Tensor(input).cuda()
            outputs = model(input)
            #print(outputs.shape)
            outputs = F.interpolate(outputs[:, :24, :, :], size=(256, 192), mode='bilinear', align_corners=True)
            target = target.cuda(non_blocking=True)

            heatmap_vis = vis_heatmaps(outputs[0, :24, :, :].detach().cpu().numpy()).astype(np.uint8)
            target_vis = vis_heatmaps(target[0].detach().cpu().numpy()).astype(np.uint8)
            print(heatmap_vis.shape, target_vis.shape)

            check = cv2.hconcat([heatmap_vis, target_vis])
            cv2.imwrite("Check.jpg", check)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target)
                for output in outputs[1:]:
                    loss += criterion(output, target)
            else:
                output = outputs
                # print(output.shape, target.shape)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: %d Loss: %.6f' % (epoch, loss * 1000))
        print("Epoch %d Finished." % epoch)
        if (epoch + 1) % 5 == 0:
            weight_file = 'ViTPose_Epoch_' + str(epoch + 1) + '.pth'
            final_model_state_file = os.path.join('result/', weight_file)
            torch.save(model.state_dict(), final_model_state_file)
if __name__ == '__main__':
    main()