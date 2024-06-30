import os
import sys
import argparse

import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import pyqtSignal

from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPen, QBrush, QColor, QPixmap, QImage

from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSlider

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
    IKResultUpdated = pyqtSignal(str)
    ConfigurationUpdated = pyqtSignal(float, float)
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

    def compute_angle(self, a, b, p):
        vec2 = np.array([a, b]) - np.array(p)
        vec1 = -np.array(p)
        alpha = np.arctan2(vec1[1], vec1[0])
        alpha = alpha + 2.0 * np.pi if alpha < 0 else alpha
        beta = 2.0 * np.pi - alpha
        mat = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
        vec2 = np.dot(mat, vec2.reshape(-1, 1))
        theta2 = np.arctan2(vec2[1], vec2[0]).item()
        theta2 = theta2 + 2.0 * np.pi if theta2 < 0 else theta2

        theta1 = np.arctan2(p[1], p[0])
        theta1 = theta1 + 2.0 * np.pi if theta1 < 0 else theta1
        print(theta1 / np.pi * 180, theta2 / np.pi * 180)

        return theta1, theta2

    def ik(self, a, b):
        if np.linalg.norm([a, b]) > link1_len + link2_len or np.linalg.norm([a, b]) < link1_len - link2_len:
            self.IKResultUpdated.emit("Position ({0}, {1}) is unreachable".format(a, b))
            return
        elif np.linalg.norm([a, b]) == link1_len + link2_len:
            x1 = link1_len / (link1_len + link2_len) * a
            y1 = link1_len / (link1_len + link2_len) * b
            p1 = (x1, y1)
            p2 = (x1, y1)
            self.IKResultUpdated.emit("One solution found at Position ({0}, {1})".format(a, b))
        elif np.linalg.norm([a, b]) == link1_len - link2_len:
            x1 = link1_len / (link1_len - link2_len) * a
            y1 = link1_len / (link1_len - link2_len) * b
            p1 = (x1, y1)
            p2 = (x1, y1)
            self.IKResultUpdated.emit("One solution found at Position ({0}, {1})".format(a, b))
        else:
            if a != 0:
                b = -b
                p1, p2 = self.solve_equation(a, b)
            else:
                b = -b
                p1, p2 = self.solve_equation(b, a)
                p1 = p1[1], p1[0]
                p2 = p2[1], p2[0]

            self.IKResultUpdated.emit("Two solution found at Position ({0}, {1})".format(a, b))

        results = [self.compute_angle(a, b, p1), self.compute_angle(a, b, p2)]

        return results

    def mouseDoubleClickEvent(self, QGraphicsSceneMouseEvent):
        scenePos = QGraphicsSceneMouseEvent.scenePos()
        a = scenePos.x()
        b = scenePos.y()

        results = self.ik(a, b)

        if results is None:
            return

        valid = False
        for theta1, theta2 in results:
            inCollision = self.update_theta(theta1 / np.pi * 180, theta2 / np.pi * 180)
            valid = not inCollision

            self.ConfigurationUpdated.emit(theta1, theta2)

            if valid:
                break

        if not valid:
            self.IKResultUpdated.emit("Collision detected in the goal state.")

    def solve_equation(self, a, b):
        k = (link2_len ** 2.0 - link1_len ** 2.0 - a ** 2.0 - b ** 2.0) / (-2.0 * a)
        A = 1 + (b / a) ** 2.0
        B = -2.0 * k * (b / a)
        C = k ** 2.0 - link1_len ** 2

        delta = B ** 2 - 4 * A * C

        y1 = (-B + np.sqrt(delta)) / (2.0 * A)
        y2 = (-B - np.sqrt(delta)) / (2.0 * A)

        if b == 0:
            i1 = np.sqrt(link1_len ** 2.0 - y1 ** 2.0)
            i2 = -np.sqrt(link1_len ** 2.0 - y2 ** 2.0)

            if np.isclose((i1 - a) ** 2.0 + (y1 - b) ** 2.0, link2_len ** 2.0):
                x1 = i1
            else:
                x1 = i2

            if np.isclose((i1 - a) ** 2.0 + (y2 - b) ** 2.0, link2_len ** 2.0):
                x2 = i1
            else:
                x2 = i2

        else:
            x1 = k - (b / a) * y1
            x2 = k - (b / a) * y2

        return (x1, y1), (x2, y2)


class CentralWidget(QWidget):
    def __init__(self, cdn_ckpt_path, use_gpu, min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height):
        super(CentralWidget, self).__init__()

        model = CollisionDetectionNetwork(input_shape=(1, 4))
        state_dict = torch.load(cdn_ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.requires_grad_(False)

        if use_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model)
        self.model = model

        self.use_gpu = use_gpu
        self.min_bbox_width = min_bbox_width
        self.max_bbox_width = max_bbox_width
        self.min_bbox_height = min_bbox_height
        self.max_bbox_height = max_bbox_height

        # generate probability map
        width = (self.min_bbox_width + self.max_bbox_width) / 2.0
        height = (self.min_bbox_height + self.max_bbox_height) / 2.0
        probs = []
        for i in np.linspace(0, np.pi * 2, 360):
            in_list = []
            for j in np.linspace(0, np.pi * 2, 360):
                in_list.append([i, j, width, height])
            input_tensor = torch.tensor(in_list, dtype=torch.float32)
            if self.use_gpu:
                input_tensor = input_tensor.cuda()
            probs.append(self.model(input_tensor).cpu().numpy().flatten())
        probs = np.array(probs)
        prob_image = probs.reshape((360, 360))
        prob_image = 1. - prob_image
        prob_image *= 255.
        prob_image = prob_image.astype(np.uint8)
        self.prob_image = np.stack([prob_image, prob_image, prob_image], axis=-1)

        self.setLayout(QHBoxLayout())

        self.conf_label = QLabel()
        pixmap = QPixmap(QImage(self.prob_image.data, self.prob_image.shape[1], self.prob_image.shape[0],
                                self.prob_image.shape[1] * 3, QImage.Format_RGB888))
        self.conf_label.setPixmap(pixmap)
        self.conf_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

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

        self.sliderLayout = QVBoxLayout(self)
        self.sliderLayout.addWidget(self.conf_label)
        self.sliderLayout.addLayout(self.theta1_layout)
        self.sliderLayout.addLayout(self.theta2_layout)
        self.sliderLayout.addLayout(self.bbox_width_layout)
        self.sliderLayout.addLayout(self.bbox_height_layout)

        self.scene = GraphicsScene()
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

        self.layout().addWidget(self.view)
        self.layout().addLayout(self.sliderLayout)

        self.update_theta()
        self.update_bbox()

        self.scene.ConfigurationUpdated.connect(self.update_slider)

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

    def update_slider(self, theta1, theta2):
        theta1 = theta1 / np.pi * 180
        theta2 = theta2 / np.pi * 180
        self.theta1_slider.setValue(theta1)
        self.theta2_slider.setValue(theta2)

    def update_theta(self):
        theta1 = self.theta1_slider.value()
        theta2 = self.theta2_slider.value()
        self.scene.update_theta(theta1, theta2)

        theta1 = int(theta1)
        theta2 = int(theta2)

        prob_image = self.prob_image.copy()
        cv2.circle(prob_image, (theta2 % 360, theta1 % 360), 3, [255, 0, 0], thickness=-1)
        pixmap = QPixmap(QImage(prob_image.data, prob_image.shape[1], prob_image.shape[0],
                                prob_image.shape[1] * 3, QImage.Format_RGB888))
        self.conf_label.setPixmap(pixmap)

    def update_bbox(self):
        width = self.bbox_width_slider.value()
        height = self.bbox_height_slider.value()
        self.scene.update_bbox(width, height)


class MainWindow(QMainWindow):
    def __init__(self, cdn_ckpt_path, use_gpu, min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height):
        super(MainWindow, self).__init__()
        self.view = CentralWidget(cdn_ckpt_path, use_gpu,
                                  min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height)
        self.setCentralWidget(self.view)
        self.statusBar().showMessage("Ready!")

        self.view.scene.IKResultUpdated.connect(self.showMessage)

    def showMessage(self, message):
        self.statusBar().showMessage(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Path Generation Network Interactive Demo.")

    # Model
    parser.add_argument("--cdn_ckpt", type=str, required=True, help="Path to the collision detection network checkpoint file.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPUs for running this script.")
    # UI
    parser.add_argument("--min_bbox_width", type=float, default=10.0, help="The minimum width of the bbox.")
    parser.add_argument("--max_bbox_width", type=float, default=40.0, help="The maximum width of the bbox.")
    parser.add_argument("--min_bbox_height", type=float, default=10.0, help="The minimum height of the bbox.")
    parser.add_argument("--max_bbox_height", type=float, default=40.0, help="The maximum height of the bbox.")
    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.cdn_ckpt):
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

    app = QApplication(sys.argv)

    mainwindow = MainWindow(args.cdn_ckpt, use_gpu,
                            args.min_bbox_width, args.max_bbox_width, args.min_bbox_height, args.max_bbox_height)
    mainwindow.setWindowTitle("Path Generation Network Interactive Demo")
    mainwindow.resize(1280, 768)
    mainwindow.show()
    app.exec_()
