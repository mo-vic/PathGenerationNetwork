import os
import sys
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
from torch.nn import functional as F

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPen, QBrush, QColor

from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5.QtWidgets import QApplication

from logger import Logger
from tensorboardX import SummaryWriter

obstacles = [
    [[511, 573], [675, 584], [620, 649], [466, 679]],
    [[634, 1148], [908, 1170], [806, 1262], [627, 1206]],
    [[690, 1004], [902, 989], [924, 1066]],
    [[828, 646], [944, 747], [903, 772], [826, 711]],
    [[426, 1085], [484, 1014], [499, 1188], [408, 1232]]
]

link1 = [658, 695]
link2 = [803, 602]
origin = [695, 898]

link1 = np.array(link1)
link2 = np.array(link2)
origin = np.array(origin)

link1 -= origin
link2 -= origin

obstacles = list(map(lambda x: x - origin, obstacles))

link1_len = np.linalg.norm(link1)
link2_len = np.linalg.norm(link2 - link1)

origin -= origin


class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None, min_bbox_width=10., max_bbox_width=40., min_bbox_height=10., max_bbox_height=40., radius=6, width=4):
        super(GraphicsScene, self).__init__(parent=parent)
        self.radius = radius
        self.width = width
        self.diameter = radius * 2

        self.min_bbox_width = min_bbox_width
        self.max_bbox_width = max_bbox_width
        self.min_bbox_height = min_bbox_height
        self.max_bbox_height = max_bbox_height

        for obstacle in obstacles:
            polygon = QGraphicsPolygonItem()
            polygonF = QPolygonF()
            for px, py in obstacle:
                polygonF.append(QPointF(px, py))
            polygon.setPolygon(polygonF)
            polygon.setPen(QPen(QColor(0, 0, 255)))
            polygon.setBrush(QBrush(QColor(0, 0, 255, 128)))
            self.addItem(polygon)

        originItem = QGraphicsEllipseItem()
        originItem.setPen(QPen(QColor(0, 0, 255)))
        originItem.setBrush(QBrush(QColor(0, 0, 255, 128)))
        originItem.setPos(QPointF(0, 0))
        originItem.setRect(origin[0] - self.radius, origin[1] - self.radius, self.diameter, self.diameter)
        self.addItem(originItem)

        self.jointItem = QGraphicsEllipseItem()
        self.jointItem.setPen(QPen(QColor(0, 0, 255)))
        self.jointItem.setBrush(QBrush(QColor(0, 0, 255, 128)))
        self.jointItem.setRect(link1[0] - self.radius, link1[1] - self.radius, self.diameter, self.diameter)
        self.addItem(self.jointItem)

        self.linkItem1 = QGraphicsLineItem()
        self.linkItem1.setPen(QPen(QColor(0, 0, 0), self.width))
        self.linkItem1.setLine(link1[0], link1[1], origin[0], origin[1])
        self.addItem(self.linkItem1)

        self.linkItem2 = QGraphicsLineItem()
        self.linkItem2.setPen(QPen(QColor(0, 0, 0), self.width))
        self.linkItem2.setLine(link2[0], link2[1], link1[0], link1[1])
        self.addItem(self.linkItem2)

        self.bboxItem = QGraphicsRectItem()
        self.bboxItem.setPen(QPen(QColor(0, 0, 0), self.width))
        self.bboxItem.setBrush(QBrush(QColor(0, 0, 0, 255)))
        self.bboxItem.setRect(-20, -20, 40, 40)
        self.addItem(self.bboxItem)

    def update_env(self, theta1, theta2, bbox_width, bbox_height):
        x1 = link1_len * np.cos(theta1)
        y1 = -link1_len * np.sin(theta1)
        self.jointItem.setRect(x1 - self.radius, y1 - self.radius, self.diameter, self.diameter)
        self.linkItem1.setLine(x1, y1, origin[0], origin[1])

        mat0 = np.array([[np.cos(theta1), np.sin(theta1), x1],
                         [-np.sin(theta1), np.cos(theta1), y1],
                         [0, 0, 1]])
        mat1 = np.array([[np.cos(-np.pi), -np.sin(-np.pi), 0],
                         [np.sin(-np.pi), np.cos(-np.pi), 0],
                         [0, 0, 1]])
        mat1 = np.dot(mat0, mat1)

        mat2 = np.array([[np.cos(theta2), -np.sin(theta2), 0],
                         [np.sin(theta2), np.cos(theta2), 0],
                         [0, 0, 1]])

        y_inv_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        x2, y2, _ = np.dot(mat1, np.dot(y_inv_mat, np.dot(mat2, (link2_len, 0, 1))))
        self.linkItem2.setLine(x2, y2, x1, y1)

        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        upper_x = bbox_width / -2.0
        upper_y = bbox_height / -2.0
        self.bboxItem.setRect(upper_x, upper_y, -2 * upper_x, -2 * upper_y)
        self.bboxItem.setRotation(angle)
        self.bboxItem.setPos(x2, y2)

        isCollide = len(self.linkItem1.collidingItems()) != 3 or len(self.linkItem2.collidingItems()) != 3 or len(self.bboxItem.collidingItems()) != 1

        self.update()

        return isCollide

    def sample(self):
        bbox_width = np.random.uniform(self.min_bbox_width, self.max_bbox_width)
        bbox_height = np.random.uniform(self.min_bbox_height, self.max_bbox_height)
        while True:
            theta1, theta2 = np.random.uniform(0.0, 2.0 * np.pi, 2)
            if self.isValid(theta1, theta2, bbox_width, bbox_height):
                start = theta1, theta2
                break

        while True:
            theta1, theta2 = np.random.uniform(0.0, 2.0 * np.pi, 2)
            if self.isValid(theta1, theta2, bbox_width, bbox_height):
                goal = theta1, theta2
                break

        return start, goal, (bbox_width, bbox_height)

    def isValid(self, theta1, theta2, bbox_width, bbox_height):
        return not self.update_env(theta1, theta2, bbox_width, bbox_height)


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


class PathGenerationNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PathGenerationNetwork, self).__init__()

        in_node = input_shape[1]
        out_node = output_shape[1]
        self.layer1 = nn.Linear(in_node, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.layer4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, out_node)

    # Returns tensor([[x0, y0, ... xn, yn]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x))) + x
        x = F.relu(self.bn3(self.layer3(x))) + x
        x = F.relu(self.bn4(self.layer4(x))) + x

        return self.layer5(x)


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
    parser = argparse.ArgumentParser(description="Train a path generation network.")

    # Data
    parser.add_argument("--min_bbox_width", type=float, default=10.0, help="The minimum width of the bbox.")
    parser.add_argument("--max_bbox_width", type=float, default=40.0, help="The maximum width of the bbox.")
    parser.add_argument("--min_bbox_height", type=float, default=10.0, help="The minimum height of the bbox.")
    parser.add_argument("--max_bbox_height", type=float, default=40.0, help="The maximum height of the bbox.")
    parser.add_argument("--datasize", type=int, default=10000, help="Size of the dataset.")
    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the collision detection network checkpoint file.")
    # Optimization
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient descent.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum of the SGD optimizer.")
    parser.add_argument("--num_curves", type=int, default=10, help="Number of curves to optimize.")
    parser.add_argument("--num_points", type=int, default=50, help="Number of points for each curve.")
    parser.add_argument("--degree", type=int, default=5, help="Degree of the Bezier curve.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for collision avoidance in loss function.")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient for curve length regularization in loss function.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Coefficient for curve segment variance in loss function.")
    parser.add_argument("--factor", type=float, default=0.2, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for measuring the new optimum, to only focus on significant changes.")
    # Misc
    parser.add_argument("--log_dir", type=str, default="../run/", help="Where to save the log?")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the log folder.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    # Check before run.
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.log_name)

    # Setting up logger
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    sys.stdout = Logger(os.path.join(log_dir, log_file))
    print(args)

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

    collision_detection_network = CollisionDetectionNetwork((1, 4))
    path_generation_network = PathGenerationNetwork((1, 6), (1, (args.degree + 1 - 2) * 2))
    state_dict = torch.load(args.ckpt)
    collision_detection_network.load_state_dict(state_dict)
    collision_detection_network.eval()
    collision_detection_network.requires_grad_(False)

    if use_gpu:
        collision_detection_network = collision_detection_network.cuda()
        collision_detection_network = torch.nn.DataParallel(collision_detection_network)

        path_generation_network = path_generation_network.cuda()
        path_generation_network = torch.nn.DataParallel(path_generation_network)

    criterion = BCELoss()
    optimizer = torch.optim.SGD(path_generation_network.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.factor,
                                                           patience=args.patience, verbose=True,
                                                           threshold=args.threshold)

    if use_gpu:
        target = torch.tensor([[0.0]] * args.batch_size * args.num_points, dtype=torch.float32).cuda()
    else:
        target = torch.tensor([[0.0]] * args.batch_size * args.num_points, dtype=torch.float32)

    video_name = os.path.join(log_dir, "training_visualization.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_name, fourcc, 30, (640, 480))

    conf_space_width = 360
    conf_space_height = 360
    mid_bbox_width = (args.min_bbox_width + args.max_bbox_width) / 2.0
    mid_bbox_height = (args.min_bbox_height + args.max_bbox_height) / 2.0

    # generate probability map
    probs = []
    for i in np.linspace(0, np.pi * 2, conf_space_height):
        in_list = []
        for j in np.linspace(0, np.pi * 2, conf_space_width):
            in_list.append([i, j, mid_bbox_width, mid_bbox_height])
        input_tensor = torch.tensor(in_list, dtype=torch.float32)
        if use_gpu:
            input_tensor = input_tensor.cuda()
        probs.append(collision_detection_network(input_tensor).cpu().numpy().flatten())
    probs = np.array(probs)
    prob_image = probs.reshape((360, 360))
    prob_image = 1. - prob_image
    prob_image *= 255.
    prob_image = prob_image.astype(np.uint8)
    conf_image = np.stack([prob_image, prob_image, prob_image], axis=-1)

    current_loss = np.inf
    best_loss = np.inf
    sep_point = 1.0 / args.degree
    spacing = np.linspace(sep_point, 1.0 - sep_point, args.degree + 1 - 2)
    spacing = np.repeat(spacing, 2, axis=-1)
    spacing = spacing.reshape(-1, 2)
    spacing = np.concatenate([spacing] * args.batch_size, axis=0)

    if use_gpu:
        spacing = torch.tensor(spacing, dtype=torch.float32).cuda()
    else:
        spacing = torch.tensor(spacing, dtype=torch.float32)

    ts = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    randomDecoder = RandomDSD(args.degree, args.num_points)
    uniformDecoder = UniformDSD(args.degree, args.num_points)

    app = QApplication(sys.argv)
    scene = GraphicsScene(min_bbox_width=args.min_bbox_width,
                          max_bbox_width=args.max_bbox_width,
                          min_bbox_height=args.min_bbox_height,
                          max_bbox_height=args.max_bbox_height)

    print("Start training...")
    start = datetime.now()

    with SummaryWriter(log_dir) as writer:
        for epoch in range(args.epochs):
            # delete and recreate app and scene to accelerate collision detection
            del app
            app = QApplication(sys.argv)
            del scene
            scene = GraphicsScene(min_bbox_width=args.min_bbox_width,
                                  max_bbox_width=args.max_bbox_width,
                                  min_bbox_height=args.min_bbox_height,
                                  max_bbox_height=args.max_bbox_height)

            all_loss = []
            all_matching_loss = []
            all_length_loss = []
            all_segment_var = []
            for idx in tqdm(range(args.datasize // args.batch_size),
                            desc="Training Epoch {0}, Loss: {1}".format(epoch, current_loss)):
                queries = []
                bboxes_info = []
                for _ in range(args.batch_size):
                    s, g, bbox_info = scene.sample()
                    query = s + g + bbox_info
                    queries.append(query)
                    bboxes_info.append(bbox_info * args.num_points)
                queries = np.array(queries, dtype=np.float32)
                queries = torch.tensor(queries)
                bboxes_info = torch.tensor(bboxes_info, dtype=torch.float32).view(-1, 2)
                if use_gpu:
                    queries = queries.cuda()
                    bboxes_info = bboxes_info.cuda()

                inter_ctrl = path_generation_network(queries)

                start_pos = queries[:, :2]
                goal_pos = queries[:, 2:4]

                inter_ctrl_init = start_pos.repeat_interleave(args.degree + 1 - 2, dim=0) + \
                                  (goal_pos - start_pos).repeat_interleave(args.degree + 1 - 2, dim=0) * spacing
                inter_ctrl_init = inter_ctrl_init.view(args.batch_size, -1)
                inter_ctrl = inter_ctrl + inter_ctrl_init

                ctrl = torch.cat([start_pos, inter_ctrl, goal_pos], dim=-1)

                if idx % 40 == 0:
                    all_data = []
                    for index in range(args.batch_size):
                        ctrl_copy = ctrl.data[index].cpu().numpy()
                        ctrl_copy = ctrl_copy.reshape(-1, 2).T
                        curve = bezier.Curve(ctrl_copy, degree=args.degree)
                        theta = curve.evaluate_multi(ts).T / np.pi * 180
                        all_data.append(theta)
                    plot_image = plot(conf_image, all_data)
                    videowriter.write(plot_image)

                optimizer.zero_grad()
                decoded_coor = randomDecoder(ctrl).view(args.batch_size * args.num_points, 2)
                out = collision_detection_network(torch.cat([decoded_coor, bboxes_info], dim=-1))
                matching_loss = criterion(out, target)

                decoded_coor = uniformDecoder(ctrl)
                pointBefore = decoded_coor[:, :-1, :]
                pointAfter = decoded_coor[:, 1:, :]
                pointDiff = pointAfter - pointBefore

                segment_length = torch.pow(torch.square(pointDiff).sum(-1), 0.5)
                curve_length = segment_length.sum(1)
                lineDist = torch.pow(torch.square(goal_pos - start_pos).sum(1), 0.5)
                length_loss = (curve_length / lineDist).mean()

                segment_mean = segment_length.mean(dim=-1).view(-1, 1)
                segment_var = torch.pow(segment_length - segment_mean, 2.0).mean()

                loss = args.alpha * matching_loss + args.beta * length_loss + args.gamma * segment_var

                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())
                all_matching_loss.append(matching_loss.item())
                all_length_loss.append(length_loss.item())
                all_segment_var.append(segment_var.item())
                writer.add_scalar("loss", loss.item(), global_step=epoch * (args.datasize // args.batch_size) + idx)
                writer.add_scalar("matching_loss", matching_loss.item(),
                                  global_step=epoch * (args.datasize // args.batch_size) + idx)
                writer.add_scalar("length_loss", length_loss.item(),
                                  global_step=epoch * (args.datasize // args.batch_size) + idx)
                writer.add_scalar("segment_var", segment_var.item(),
                                  global_step=epoch * (args.datasize // args.batch_size) + idx)

            current_loss = np.mean(all_loss).item()
            scheduler.step(current_loss)

            writer.add_scalar("all_loss", np.mean(all_loss).item(), global_step=epoch)
            writer.add_scalar("all_matching_loss", np.mean(all_matching_loss).item(), global_step=epoch)
            writer.add_scalar("all_length_loss", np.mean(all_length_loss).item(), global_step=epoch)
            writer.add_scalar("all_segment_var", np.mean(all_segment_var).item(), global_step=epoch)

            if current_loss < best_loss:
                best_loss = current_loss

                if use_gpu:
                    torch.save(path_generation_network.module.state_dict(), os.path.join(log_dir, "%04d.pth" % epoch))
                else:
                    torch.save(path_generation_network.state_dict(), os.path.join(log_dir, "%04d.pth" % epoch))

    videowriter.release()

    if use_gpu:
        torch.save(path_generation_network.module.state_dict(), os.path.join(log_dir, "final.pth"))
    else:
        torch.save(path_generation_network.state_dict(), os.path.join(log_dir, "final.pth"))

    elapsed_time = str(datetime.now() - start)
    print("Finish training. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
