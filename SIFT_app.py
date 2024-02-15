#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)
		
        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

		# Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                        bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        img = cv2.imread("/home/fizzer/Downloads/000_image.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypt1, desc1 = sift.detectAndCompute(img, None)
        keypt2, desc2 = sift.detectAndCompute(frame, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(desc1, desc2)

        matches = sorted(matches, key = lambda x:x.distance)
        matched_img = cv2.drawMatches(img, keypt1, frame, keypt2, matches[:50], frame, flags=2)
        pixmap = self.convert_cv_to_pixmap(matched_img)
        self.live_image_label.setPixmap(pixmap)
    
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

    def run_SIFT(self, frame):
        img = cv2.imread("000_image.jpg", cv2.IMREAD_GRAYSCALE)

        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img, None)
        index = dict(algorithm=0, trees=5)
        search = dict()
        flann = cv2.FlannBasedMatcher(index, search)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_gray, desc_gray = sift.detectAndCompute(gray, None)
        matches = flann.knnMatch(desc_image, desc_gray, k=2)
        good_pts = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_pts.append(m)

        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_pts]).reshape(-1,1,2)
        train_pts = np.float32([kp_gray[m.trainIdx].pt for m in good_pts]).reshape(-1,1,2)
        mtx, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        match_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, mtx)

        return cv2.polylines(frame, [np.int32(dst)], True, (255,0,0), 3)
        

        


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
