import os
import sys
import argparse

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPen, QBrush, QColor

from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5.QtWidgets import QApplication


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
        upper_x = np.random.uniform(-self.max_bbox_width / 2.0, -self.min_bbox_width / 2.0)
        upper_y = np.random.uniform(-self.max_bbox_height / 2.0, -self.min_bbox_height / 2.0)
        self.bboxItem.setRect(upper_x, upper_y, -2 * upper_x, -2 * upper_y)
        self.bboxItem.setRotation(angle)
        self.bboxItem.setPos(x2, y2)

        isCollide = len(self.linkItem1.collidingItems()) != 3 or len(self.linkItem2.collidingItems()) != 3 or len(self.bboxItem.collidingItems()) != 1
        if isCollide:
            self.bboxItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        else:
            self.bboxItem.setBrush(QBrush(QColor(0, 0, 0, 255)))

        self.update()

        return isCollide, (-2 * upper_x, -2 * upper_y)


class CentralWidget(QWidget):
    def __init__(self, num_train_samples, num_test_samples, save_path,
                 min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height):
        super(CentralWidget, self).__init__()
        self.setLayout(QHBoxLayout())

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.save_path = save_path

        self.trainset = []
        self.testset = []

        self.train_samples_counter = 0
        self.test_samples_counter = 0

        self.num_free = 0
        self.num_collision = 0

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

        self.sliderLayout = QVBoxLayout(self)
        self.sliderLayout.addLayout(self.theta1_layout)
        self.sliderLayout.addLayout(self.theta2_layout)

        self.scene = GraphicsScene(None, min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

        self.layout().addWidget(self.view)
        self.layout().addLayout(self.sliderLayout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.change_configuraton)
        self.timer.setInterval(0)

    def update_theta1(self, value):
        self.theta1_label.setText("theta1=%d" % value)
        self.update_theta()

    def update_theta2(self, value):
        self.theta2_label.setText("theta2=%d" % value)
        self.update_theta()

    def update_theta(self):
        theta1 = self.theta1_slider.value()
        theta2 = self.theta2_slider.value()
        self.scene.update_theta(theta1, theta2)

    def change_configuraton(self):
        if self.train_samples_counter < self.num_train_samples or self.test_samples_counter < self.num_test_samples:
            theta1 = np.random.uniform(0, 360)
            theta2 = np.random.uniform(0, 360)
            isCollide, (bbox_width, bbox_height) = self.scene.update_theta(theta1, theta2)

            if isCollide:
                self.num_collision += 1
            else:
                self.num_free += 1

            sample = [theta1, theta2, bbox_width, bbox_height, int(isCollide)]
            if self.train_samples_counter < self.num_train_samples:
                self.train_samples_counter += 1
                self.trainset.append(sample)
                if self.train_samples_counter % 2000 == 0:
                    print("{0} out of {1} training samples generated.".format(self.train_samples_counter, self.num_train_samples))
            else:
                self.test_samples_counter += 1
                self.testset.append(sample)
                if self.test_samples_counter % 2000 == 0:
                    print("{0} out of {1} testing samples generated.".format(self.test_samples_counter, self.num_test_samples))

        else:
            self.timer.stop()

            print("total number of samples:", self.num_collision + self.num_free)
            print("number of samples in free configuration space:", self.num_free)
            print("number of samples in obstacle space:", self.num_collision)
            print("free / collision ratio:", self.num_free / self.num_collision)

            os.mkdir(self.save_path)

            np.save(os.path.join(self.save_path, "train.npy"), np.array(self.trainset))
            np.save(os.path.join(self.save_path, "test.npy"), np.array(self.testset))


class MainWindow(QMainWindow):
    def __init__(self, num_train_samples, num_test_samples, save_path,
                 min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height):
        super(MainWindow, self).__init__()
        self.view = CentralWidget(num_train_samples, num_test_samples, save_path,
                                  min_bbox_width, max_bbox_width, min_bbox_height, max_bbox_height)
        self.view.timer.start()
        self.setCentralWidget(self.view)
        self.statusBar().showMessage("Ready!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Collision Data.")

    parser.add_argument("--num_train_samples", type=int, default=100000, help="The number of training samplings.")
    parser.add_argument("--num_test_samples", type=int, default=10000, help="The number of testing samplings.")
    parser.add_argument("--min_bbox_width", type=float, default=10.0, help="The minimum width of the bbox.")
    parser.add_argument("--max_bbox_width", type=float, default=40.0, help="The maximum width of the bbox.")
    parser.add_argument("--min_bbox_height", type=float, default=10.0, help="The minimum height of the bbox.")
    parser.add_argument("--max_bbox_height", type=float, default=40.0, help="The maximum height of the bbox.")
    parser.add_argument("--save_path", type=str, default="../data", help="Path to save the collision data.")

    args = parser.parse_args()
    print(args)

    assert not os.path.exists(args.save_path)

    app = QApplication(sys.argv)

    mainwindow = MainWindow(args.num_train_samples, args.num_test_samples, args.save_path,
                            args.min_bbox_width, args.max_bbox_width,
                            args.min_bbox_height, args.max_bbox_height)
    mainwindow.setWindowTitle("Generate Collision Data")
    mainwindow.resize(1280, 768)
    mainwindow.show()
    app.exec_()
