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


img = cv2.imread('aa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

nPatternFound = 0

if ret == True:
    corners1 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (nCols, nRows), corners1, ret)
    cv2.imshow('img', img)
    imgpoints.append(corners)
    objpoints.append(objp)
    nPatternFound += 1
    print(nPatternFound, " Calibration Pattern Bulundu")
    # cv2.waitKey(0)


if (nPatternFound > 0):
    newimg = cv2.imread('calib.jpg')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (newimg.shape[1], newimg.shape[0]), None, None)
    cv2.imshow('newimg', newimg)

    print(mtx)
    print(dist)
    h, w = newimg.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(newimg, mtx, dist, None, newcameramtx)
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # dst = cv2.remap(newimg, mapx, mapy, cv2.INTER_LINEAR)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv2.imwrite("UndistortedImage.png", dst)
    cv2.imshow('UndistortedImage', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()