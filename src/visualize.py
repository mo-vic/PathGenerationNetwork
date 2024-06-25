import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPen, QBrush, QColor

from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QPushButton

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5.QtWidgets import QApplication

import torch
from torch.backends import cudnn

from train_collision_detection_network import CollisionDetectionNetwork


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
    def __init__(self, parent=None, radius=6, width=4):
        super(GraphicsScene, self).__init__(parent=parent)
        self.radius = radius
        self.width = width
        self.diameter = radius * 2

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

    def update_theta(self, theta1, theta2):
        theta1 = theta1 / 180 * np.pi
        theta2 = theta2 / 180 * np.pi

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
        self.bboxItem.setRotation(angle)
        self.bboxItem.setPos(x2, y2)

        isCollide = len(self.linkItem1.collidingItems()) != 3 or len(self.linkItem2.collidingItems()) != 3 or len(self.bboxItem.collidingItems()) != 1
        if isCollide:
            self.bboxItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        else:
            self.bboxItem.setBrush(QBrush(QColor(0, 0, 0, 255)))

        self.update()

        return isCollide

    def update_bbox(self, width, height):
        upper_x = -width / 2.0
        upper_y = -height / 2.0

        self.bboxItem.setRect(upper_x, upper_y, -2 * upper_x, -2 * upper_y)

        isCollide = len(self.linkItem1.collidingItems()) != 3 or len(self.linkItem2.collidingItems()) != 3 or len(self.bboxItem.collidingItems()) != 1
        if isCollide:
            self.bboxItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        else:
            self.bboxItem.setBrush(QBrush(QColor(0, 0, 0, 255)))

        self.update()


class CentralWidget(QWidget):
    def __init__(self, ckpt_path, save_path, use_gpu,
                 min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height,
                 map_width, map_height):
        super(CentralWidget, self).__init__()

        model = CollisionDetectionNetwork(input_shape=(1, 4))
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.requires_grad_(False)

        if use_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model)
        self.model = model

        self.use_gpu = use_gpu
        self.save_path = save_path
        self.min_bbox_width = min_bbox_width
        self.max_bbox_width = max_bbox_width
        self.min_bbox_height = min_bbox_height
        self.max_bbox_height = max_bbox_height
        self.map_width = map_width
        self.map_height = map_height

        self.counter = 0
        self.gt_map = np.zeros((map_height, map_width), np.uint8)
        gt_x_coor = np.linspace(0.0, 2.0 * np.pi, map_width)
        gt_y_coor = np.linspace(0.0, 2.0 * np.pi, map_height)
        self.gt_sample_grid = np.meshgrid(gt_x_coor, gt_y_coor)

        self.timer = QTimer()
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.draw_configuration_space)

        self.setLayout(QHBoxLayout())

        self.theta1_slider = QSlider()
        self.theta1_slider.setMinimum(0)
        self.theta1_slider.setMaximum(360)
        self.theta1_slider.setValue(0)
        self.theta1_slider.setOrientation(Qt.Horizontal)
        self.theta1_slider.valueChanged.connect(self.update_theta1)
        self.theta1_label = QLabel("theta1=%d" % 0)
        self.theta1_layout = QHBoxLayout(self)
        self.theta1_layout.addWidget(self.theta1_slider)
        self.theta1_layout.addWidget(self.theta1_label)

        self.theta2_slider = QSlider()
        self.theta2_slider.setMinimum(0)
        self.theta2_slider.setMaximum(360)
        self.theta2_slider.setValue(0)
        self.theta2_slider.setOrientation(Qt.Horizontal)
        self.theta2_slider.valueChanged.connect(self.update_theta2)
        self.theta2_label = QLabel("theta2=%d" % 0)
        self.theta2_layout = QHBoxLayout(self)
        self.theta2_layout.addWidget(self.theta2_slider)
        self.theta2_layout.addWidget(self.theta2_label)

        self.bbox_width_slider = QSlider()
        self.bbox_width_slider.setMinimum(self.min_bbox_width)
        self.bbox_width_slider.setMaximum(self.max_bbox_width)
        self.bbox_width_slider.setValue(self.min_bbox_width)
        self.bbox_width_slider.setOrientation(Qt.Horizontal)
        self.bbox_width_slider.valueChanged.connect(self.update_bbox_width)
        self.bbox_width_label = QLabel("bbox width=%d" % self.min_bbox_width)
        self.bbox_width_layout = QHBoxLayout(self)
        self.bbox_width_layout.addWidget(self.bbox_width_slider)
        self.bbox_width_layout.addWidget(self.bbox_width_label)

        self.bbox_height_slider = QSlider()
        self.bbox_height_slider.setMinimum(self.min_bbox_height)
        self.bbox_height_slider.setMaximum(self.max_bbox_height)
        self.bbox_height_slider.setValue(self.min_bbox_height)
        self.bbox_height_slider.setOrientation(Qt.Horizontal)
        self.bbox_height_slider.valueChanged.connect(self.update_bbox_height)
        self.bbox_height_label = QLabel("bbox height=%d" % self.min_bbox_height)
        self.bbox_height_layout = QHBoxLayout(self)
        self.bbox_height_layout.addWidget(self.bbox_height_slider)
        self.bbox_height_layout.addWidget(self.bbox_height_label)

        self.evaluation_button = QPushButton("Evaluate Network Output")
        self.evaluation_button.clicked.connect(self.evaluate)
        self.draw_conf_space_button = QPushButton("Generate Ground Truth Configuration Space")
        self.draw_conf_space_button.clicked.connect(lambda: self.timer.start())

        self.sliderLayout = QVBoxLayout(self)
        self.sliderLayout.addLayout(self.theta1_layout)
        self.sliderLayout.addLayout(self.theta2_layout)
        self.sliderLayout.addLayout(self.bbox_width_layout)
        self.sliderLayout.addLayout(self.bbox_height_layout)
        self.sliderLayout.addWidget(self.evaluation_button)
        self.sliderLayout.addWidget(self.draw_conf_space_button)

        self.scene = GraphicsScene()
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

        self.layout().addWidget(self.view)
        self.layout().addLayout(self.sliderLayout)

        self.update_theta()
        self.update_bbox()

    def update_theta1(self, value):
        self.theta1_label.setText("theta1=%d" % value)
        self.update_theta()

    def update_theta2(self, value):
        self.theta2_label.setText("theta2=%d" % value)
        self.update_theta()

    def update_bbox_width(self, value):
        self.bbox_width_label.setText("bbox width=%d" % value)
        self.update_bbox()

    def update_bbox_height(self, value):
        self.bbox_height_label.setText("bbox height=%d" % value)
        self.update_bbox()

    def update_theta(self):
        theta1 = self.theta1_slider.value()
        theta2 = self.theta2_slider.value()
        self.scene.update_theta(theta1, theta2)

    def update_bbox(self):
        width = self.bbox_width_slider.value()
        height = self.bbox_height_slider.value()
        self.scene.update_bbox(width, height)

    def evaluate(self):
        self.evaluation_button.setEnabled(False)
        theta1 = self.theta1_slider.value()
        theta2 = self.theta2_slider.value()
        theta1, theta2 = theta1 / 180 * np.pi, theta2 / 180 * np.pi

        in_list = []
        size_in = np.linspace(self.min_bbox_width, self.max_bbox_width, 256)
        for width in size_in:
            in_list.append([theta1, theta2, width, self.min_bbox_height])
        input_tensor = torch.tensor(in_list, dtype=torch.float32)
        if self.use_gpu:
            input_tensor = input_tensor.cuda()
        width_out = self.model(input_tensor).cpu().numpy().flatten()

        in_list.clear()
        size_in = np.linspace(self.min_bbox_height, self.max_bbox_height, 256)
        for height in size_in:
            in_list.append([theta1, theta2, self.min_bbox_width, height])
        input_tensor = torch.tensor(in_list, dtype=torch.float32)
        if self.use_gpu:
            input_tensor = input_tensor.cuda()
        height_out = self.model(input_tensor).cpu().numpy().flatten()

        in_list.clear()
        size_in = np.linspace(np.sqrt(self.min_bbox_height ** 2.0 + self.min_bbox_width ** 2.0),
                              np.sqrt(self.max_bbox_height ** 2.0 + self.max_bbox_width ** 2.0), 256)
        for diag in size_in:
            in_list.append([theta1, theta2, diag, diag])
        input_tensor = torch.tensor(in_list, dtype=torch.float32)
        if self.use_gpu:
            input_tensor = input_tensor.cuda()
        diag_out = self.model(input_tensor).cpu().numpy().flatten()

        fig, ax = plt.subplots()
        ax.set_xlabel("size")
        ax.set_ylabel("probability")

        plt.ylim(0.0, 1.0)

        plt.plot(size_in, width_out, color="C0", label="horizontal")
        plt.plot(size_in, height_out, color="C1", label="vertical")
        plt.plot(size_in, diag_out, color="C2", label="diagonal")
        plt.legend()

        save_path = os.path.join(self.save_path, "bbox")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_filename = os.path.join(save_path, "prob_in_size.pdf")
        plt.savefig(save_filename, bbox_inches="tight", dpi=200)
        plt.close("all")

        # generate probability map
        width = self.bbox_width_slider.value()
        height = self.bbox_height_slider.value()
        probs = []
        for i in np.linspace(0, np.pi * 2, self.map_height):
            in_list = []
            for j in np.linspace(0, np.pi * 2, self.map_width):
                in_list.append([i, j, width, height])
            input_tensor = torch.tensor(in_list, dtype=torch.float32)
            if self.use_gpu:
                input_tensor = input_tensor.cuda()
            probs.append(self.model(input_tensor).cpu().numpy().flatten())
        probs = np.array(probs)
        prob_image = probs.reshape((self.map_height, self.map_width))

        save_path = os.path.join(self.save_path, "conf_space")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, ax = plt.subplots()
        ax.set_xlabel("$\\theta_2$")
        ax.set_ylabel("$\\theta_1$")
        # plt.set_x_tick
        xticks = np.array([0, 90, 180, 270, 360]) * self.map_width / 360.0
        xticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

        yticks = np.array([0, 90, 180, 270, 360]) * self.map_height / 360.0
        yticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

        plt.xticks(xticks, xticks_label)
        plt.yticks(yticks, yticks_label)

        plt.imshow(prob_image, cmap="jet")

        plt.savefig(os.path.join(save_path, "prob_map.pdf"), bbox_inches="tight", dpi=200)
        plt.close("all")

        self.evaluation_button.setEnabled(True)

    def draw_configuration_space(self):
        self.draw_conf_space_button.setEnabled(False)
        if self.counter < self.map_width * self.map_height:
            row = self.counter % self.map_height
            col = int(self.counter / self.map_height)

            theta2 = self.gt_sample_grid[0][row, col] / np.pi * 180
            theta1 = self.gt_sample_grid[1][row, col] / np.pi * 180

            isCollide = int(self.scene.update_theta(theta1, theta2))

            self.gt_map[row, col] = (1 - isCollide) * 255
            self.counter += 1
        else:
            self.counter = 0
            self.timer.stop()

            save_path = os.path.join(self.save_path, "conf_space")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            gt_map = np.stack([self.gt_map] * 3, axis=-1)

            fig, ax = plt.subplots()
            ax.set_xlabel("$\\theta_2$")
            ax.set_ylabel("$\\theta_1$")
            # plt.set_x_tick
            xticks = np.array([0, 90, 180, 270, 360]) * self.map_width / 360.0
            xticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

            yticks = np.array([0, 90, 180, 270, 360]) * self.map_height / 360.0
            yticks_label = [0, "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]

            plt.xticks(xticks, xticks_label)
            plt.yticks(yticks, yticks_label)

            plt.imshow(gt_map)

            plt.savefig(os.path.join(save_path, "gt_map.pdf"), bbox_inches="tight", dpi=200)
            plt.close("all")

            self.draw_conf_space_button.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self, ckpt_path, save_path, use_gpu, min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height,
                 map_width, map_height):
        super(MainWindow, self).__init__()
        self.view = CentralWidget(ckpt_path, save_path, use_gpu,
                                  min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height,
                                  map_width, map_height)
        self.setCentralWidget(self.view)
        self.statusBar().showMessage("Ready!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizer Network Output.")

    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPUs for running this script.")
    # Figure
    parser.add_argument("--map_width", type=int, default=360, help="Width of the gt map and prob map.")
    parser.add_argument("--map_height", type=int, default=360, help="Height of the gt map and prob map.")
    parser.add_argument("--save_path", type=str, default="../diagram/visualization",
                        help="Path to save the figure.")
    # UI
    parser.add_argument("--min_bbox_width", type=float, default=10.0, help="The minimum width of the bbox.")
    parser.add_argument("--max_bbox_width", type=float, default=40.0, help="The maximum width of the bbox.")
    parser.add_argument("--min_bbox_height", type=float, default=10.0, help="The minimum height of the bbox.")
    parser.add_argument("--max_bbox_height", type=float, default=40.0, help="The maximum height of the bbox.")
    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    print(args)

    assert args.map_width > 0
    assert args.map_height > 0

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

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    app = QApplication(sys.argv)

    mainwindow = MainWindow(args.ckpt, args.save_path, use_gpu,
                            args.min_bbox_width, args.max_bbox_width, args.min_bbox_height, args.max_bbox_height,
                            args.map_width, args.map_height)
    mainwindow.setWindowTitle("Visualizer Network Output")
    mainwindow.resize(1280, 768)
    mainwindow.show()
    app.exec_()
