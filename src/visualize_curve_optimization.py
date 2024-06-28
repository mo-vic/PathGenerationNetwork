import os
import argparse
from datetime import datetime

import cv2
import bezier
import numpy as np
from tqdm import tqdm
from scipy.special import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import BCELoss
from torch.backends import cudnn
from torch.autograd import Variable
from torch.nn import functional as F


# Differentiable Curve Decoder
class RandomDSD(torch.nn.Module):
    def __init__(self, degree, num_points):
        super(RandomDSD, self).__init__()
        self.degree = degree
        self.num_points = num_points

        c = np.zeros((1, self.degree + 1), dtype=np.float32)
        for i in range(0, self.degree + 1):
            c[0, i] = comb(self.degree, i)

        self.c = c

    def forward(self, inputs):
        # Get the device info
        device = inputs.device
        num_curves = inputs.size(0)

        # random sampled ts
        ts = torch.tensor(np.random.uniform(0.0, 1.0, num_curves * self.num_points), dtype=torch.float32,
                          device=device)

        ts = ts.view((-1, 1)).repeat([1, self.degree + 1])
        c = torch.from_numpy(self.c.copy()).to(device)
        pow1 = torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        pow2 = self.degree - torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        ts = c * torch.pow(ts, pow1) * torch.pow(1 - ts, pow2)

        x_ctrls = inputs[:, 0::2]
        y_ctrls = inputs[:, 1::2]
        x_ctrls = x_ctrls.view((-1, self.degree + 1))
        y_ctrls = y_ctrls.view((-1, self.degree + 1))
        x_ctrls = x_ctrls.repeat_interleave(self.num_points, dim=0)
        y_ctrls = y_ctrls.repeat_interleave(self.num_points, dim=0)
        decoded_x = (ts * x_ctrls).sum(dim=-1)
        decoded_y = (ts * y_ctrls).sum(dim=-1)

        decoded_x = decoded_x.unsqueeze(1)
        decoded_y = decoded_y.unsqueeze(1)

        # the arrangement of decoded_coor is [[x, y], [x, y]]
        # the arrangement of the output of bezier.Curve.evaluate is [[x, x, ...], [y, y...]]
        decoded_coor = torch.cat([decoded_x, decoded_y], dim=1)
        decoded_coor = decoded_coor.view(num_curves, self.num_points, 2)

        return decoded_coor


class UniformDSD(torch.nn.Module):
    def __init__(self, degree, num_points):
        super(UniformDSD, self).__init__()
        self.degree = degree
        self.num_points = num_points
        c = np.zeros((1, self.degree + 1), dtype=np.float32)
        for i in range(0, self.degree + 1):
            c[0, i] = comb(self.degree, i)

        self.c = c

    def forward(self, inputs):
        # Get the device info
        device = inputs.device
        num_curves = inputs.size(0)

        # uniformly distributed ts
        ts = torch.tensor([np.linspace(0.0, 1.0, self.num_points)] * num_curves, dtype=torch.float32,
                          device=device)

        ts = ts.view((-1, 1)).repeat([1, self.degree + 1])
        c = torch.from_numpy(self.c.copy()).to(device)
        pow1 = torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        pow2 = self.degree - torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        ts = c * torch.pow(ts, pow1) * torch.pow(1 - ts, pow2)

        x_ctrls = inputs[:, 0::2]
        y_ctrls = inputs[:, 1::2]
        x_ctrls = x_ctrls.view((-1, self.degree + 1))
        y_ctrls = y_ctrls.view((-1, self.degree + 1))
        x_ctrls = x_ctrls.repeat_interleave(self.num_points, dim=0)
        y_ctrls = y_ctrls.repeat_interleave(self.num_points, dim=0)
        decoded_x = (ts * x_ctrls).sum(dim=-1)
        decoded_y = (ts * y_ctrls).sum(dim=-1)

        decoded_x = decoded_x.unsqueeze(1)
        decoded_y = decoded_y.unsqueeze(1)

        # the arrangement of decoded_coor is [[x, y], [x, y]]
        # the arrangement of the output of bezier.Curve.evaluate is [[x, x, ...], [y, y...]]
        decoded_coor = torch.cat([decoded_x, decoded_y], dim=1)
        decoded_coor = decoded_coor.view(num_curves, self.num_points, 2)

        return decoded_coor


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

        return torch.sigmoid(self.layer5(x))


def build_model(input_shape):
    model = CollisionDetectionNetwork(input_shape)

    return model


def plot(conf_image, all_data):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel("$\\theta_2$")
    ax.set_ylabel("$\\theta_1$")

    # plt.set_x_tick
    xticks = np.array([0, 90, 180, 270, 360])
    xticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

    yticks = np.array([0, 90, 180, 270, 360])
    yticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

    plt.xticks(xticks, xticks_label)
    plt.yticks(yticks, yticks_label)

    plt.imshow(conf_image)

    for data_idx, data in enumerate(all_data):
        plt.plot(data[1:-1, 1], data[1:-1, 0], color="C{0}".format(data_idx % 10), ls="--")

        plt.scatter(data[0, 1], data[0, 0], color="C{0}".format(data_idx % 10), s=12, zorder=len(all_data) + 1)
        plt.scatter(data[-1, 1], data[-1, 0], color="C{0}".format(data_idx % 10), s=12, zorder=len(all_data) + 1)

    fig.canvas.draw()
    plot_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape([480, 640, 3])
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

    plt.close("all")

    return plot_image


def main():
    parser = argparse.ArgumentParser(description="Visualize Curve Optimization.")

    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    # Input
    parser.add_argument("--bbox_width", type=float, default=25.0, help="The width of the bbox.")
    parser.add_argument("--bbox_height", type=float, default=25.0, help="The height of the bbox.")
    # Optimization
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient descent.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum of the SGD optimizer.")
    parser.add_argument("--num_curves", type=int, default=10, help="Number of curves to optimize.")
    parser.add_argument("--num_points", type=int, default=50, help="Number of points for each curve.")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the Bezier curve.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for collision avoidance in loss function.")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient for curve length regularization in loss function.")
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

    sep_point = 1.0 / args.degree
    spacing = np.linspace(sep_point, 1.0 - sep_point, args.degree + 1 - 2)
    spacing = np.repeat(spacing, 2, axis=-1)
    spacing = spacing.reshape(-1, 2)
    spacing = np.concatenate([spacing] * args.num_curves, axis=0)

    if use_gpu:
        bbox_info = torch.tensor([[args.bbox_width, args.bbox_height]] * (args.num_points * args.num_curves), dtype=torch.float32).cuda()
        start_pos = torch.tensor(np.random.uniform(0, 2.0 * np.pi, (args.num_curves, 2)), dtype=torch.float32).cuda()
        goal_pos = torch.tensor(np.random.uniform(0, 2.0 * np.pi, (args.num_curves, 2)), dtype=torch.float32).cuda()
        inter_ctrl_init = start_pos.repeat_interleave(args.degree + 1 - 2, dim=0) + \
                          (goal_pos - start_pos).repeat_interleave(args.degree + 1 - 2, dim=0) * \
                          torch.tensor(spacing, dtype=torch.float32).cuda()
        inter_ctrl_init = inter_ctrl_init.view(args.num_curves, -1)
        inter_ctrl = Variable(inter_ctrl_init).cuda()
    else:
        bbox_info = torch.tensor([[args.bbox_width, args.bbox_height]] * (args.num_points * args.num_curves), dtype=torch.float32)
        start_pos = torch.tensor(np.random.uniform(0, 2.0 * np.pi, (args.num_curves, 2)), dtype=torch.float32)
        goal_pos = torch.tensor(np.random.uniform(0, 2.0 * np.pi, (args.num_curves, 2)), dtype=torch.float32)
        inter_ctrl_init = start_pos.repeat_interleave(args.degree + 1 - 2, dim=0) + \
                          (goal_pos - start_pos).repeat_interleave(args.degree + 1 - 2, dim=0) * \
                          torch.tensor(spacing, dtype=torch.float32)
        inter_ctrl_init = inter_ctrl_init.view(args.num_curves, -1)
        inter_ctrl = Variable(inter_ctrl_init)
    inter_ctrl.requires_grad_(True)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    criterion = BCELoss()
    optimizer = torch.optim.SGD([inter_ctrl], lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.factor,
                                                           patience=args.patience, verbose=True,
                                                           threshold=args.threshold)

    if use_gpu:
        target = torch.tensor([[0.0]] * args.num_curves * args.num_points, dtype=torch.float32).cuda()
    else:
        target = torch.tensor([[0.0]] * args.num_curves * args.num_points, dtype=torch.float32)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    video_file = os.path.join(args.save_path, "CurveOptimization.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))

    conf_space_width = 360
    conf_space_height = 360

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
    conf_image = np.stack([prob_image, prob_image, prob_image], axis=-1)

    ts = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    randomDecoder = RandomDSD(args.degree, args.num_points)
    uniformDecoder = UniformDSD(args.degree, args.num_points)

    print("Start optimizing...")
    start = datetime.now()

    for it in tqdm(range(args.num_iter)):
        ctrl = torch.cat([start_pos, inter_ctrl, goal_pos], dim=-1)

        all_data = []
        for idx in range(args.num_curves):
            ctrl_copy = ctrl.data[idx].cpu().numpy()
            ctrl_copy = ctrl_copy.reshape(-1, 2).T
            curve = bezier.Curve(ctrl_copy, degree=args.degree)
            theta = curve.evaluate_multi(ts).T / np.pi * 180
            all_data.append(theta)

        plot_image = plot(conf_image, all_data)
        videowriter.write(plot_image)

        optimizer.zero_grad()
        decoded_coor = randomDecoder(ctrl).view(args.num_curves * args.num_points, 2)
        out = model(torch.cat([decoded_coor, bbox_info], dim=-1))
        matching_loss = criterion(out, target)

        decoded_coor = uniformDecoder(ctrl)
        pointBefore = decoded_coor[:, :-1, :]
        pointAfter = decoded_coor[:, 1:, :]
        pointDiff = pointAfter - pointBefore

        curve_length = torch.pow(torch.square(pointDiff).sum(-1), 0.5).sum(1)
        lineDist = torch.pow(torch.square(goal_pos - start_pos).sum(1), 0.5)
        length_loss = (curve_length / lineDist).mean()

        loss = args.alpha * matching_loss + args.beta * length_loss

        loss.backward()
        optimizer.step()

        if it % 20 == 0:
            scheduler.step(loss.item())

    videowriter.release()

    elapsed_time = str(datetime.now() - start)
    print("Finish optimizing. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
