import numpy as np
import cv2
import glob

nRows = 6
nCols = 7
dimension = 25

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

objp = np.zeros((nRows*nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

objpoints = []
imgpoints = []


img = cv2.imread('333.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

if ret == True:
    corners1 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (nCols, nRows), corners1, ret)
    cv2.imshow('img', img)
    imgpoints.append(corners)
    objpoints.append(objp)
    cv2.waitKey(0)
cv2.destroyAllWindows()