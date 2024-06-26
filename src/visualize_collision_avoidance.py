import os
import argparse
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import BCELoss
from torch.backends import cudnn
from torch.autograd import Variable
from torch.nn import functional as F


class CollisionDetectionNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CollisionDetectionNetwork, self).__init__()

        in_node = input_shape[1]
        self.layer1 = nn.Linear(in_node, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.layer4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x))) + x
        x = F.relu(self.bn3(self.layer3(x))) + x
        x = F.relu(self.bn4(self.layer4(x))) + x

        return F.sigmoid(self.layer5(x))


def build_model(input_shape):
    model = CollisionDetectionNetwork(input_shape)

    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize Collision Avoidance.")

    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    # Input
    parser.add_argument("--bbox_width", type=float, default=25.0, help="The width of the bbox.")
    parser.add_argument("--bbox_height", type=float, default=25.0, help="The height of the bbox.")
    # Optimization
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for gradient descent.")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum of the SGD optimizer.")
    parser.add_argument("--factor", type=float, default=0.2, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for measuring the new optimum, to only focus on significant changes.")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of iterations.")
    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--save_path", type=str, default="../video", help="Path to save the video.")

    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError

    for s in args.gpu_ids:
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id:{}".format(s))
            raise ValueError

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    if args.gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)
        else:
            use_gpu = False
    else:
        use_gpu = False

    torch.manual_seed(args.seed)

    model = build_model((1, 4))
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)

    if use_gpu:
        theta = Variable(torch.tensor(np.random.uniform(0, 2.0 * np.pi, (1, 2)), dtype=torch.float32)).cuda()
        bbox_info = torch.tensor([[args.bbox_width, args.bbox_height]], dtype=torch.float32).cuda()
    else:
        theta = Variable(torch.tensor(np.random.uniform(0, 2.0 * np.pi, (1, 2)), dtype=torch.float32))
        bbox_info = torch.tensor([[args.bbox_width, args.bbox_height]], dtype=torch.float32)
    theta.requires_grad_(True)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    criterion = BCELoss()
    optimizer = torch.optim.SGD([theta], lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.factor,
                                                           patience=args.patience, verbose=True,
                                                           threshold=args.threshold)

    if use_gpu:
        target = torch.tensor([[0.0]], dtype=torch.float32).cuda()
    else:
        target = torch.tensor([[0.0]], dtype=torch.float32)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_file = os.path.join(args.save_path, "CollisionAvoidance.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))

    conf_space_width = 360
    conf_space_height = 360
    offset_h = (480 - conf_space_width) // 2
    offset_w = (640 - conf_space_height) // 2

    # generate probability map
    probs = []
    for i in np.linspace(0, np.pi * 2, conf_space_height):
        in_list = []
        for j in np.linspace(0, np.pi * 2, conf_space_width):
            in_list.append([i, j, args.bbox_width, args.bbox_height])
        input_tensor = torch.tensor(in_list, dtype=torch.float32)
        if use_gpu:
            input_tensor = input_tensor.cuda()
        probs.append(model(input_tensor).cpu().numpy().flatten())
    probs = np.array(probs)
    prob_image = probs.reshape((360, 360))
    prob_image = 1. - prob_image
    prob_image *= 255.
    prob_image = prob_image.astype(np.uint8)
    prob_image = np.stack([prob_image, prob_image, prob_image], axis=-1)
    image = np.pad(prob_image, [[offset_h, offset_h], [offset_w, offset_w], [0, 0]])

    print("Start optimizing...")
    start = datetime.now()

    for it in tqdm(range(args.num_iter)):
        theta1, theta2 = theta.data.cpu().numpy().flatten() / np.pi * 180
        theta1 += offset_h
        theta2 += offset_w

        image_copy = image.copy()
        cv2.circle(image_copy, (int(theta2), int(theta1)), 3, [0, 0, 255], thickness=-1)
        videowriter.write(image_copy)

        optimizer.zero_grad()
        out = model(torch.cat([theta, bbox_info], dim=-1))
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if it % 20 == 0:
            scheduler.step(loss.item())

    videowriter.release()

    elapsed_time = str(datetime.now() - start)
    print("Finish optimizing. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
