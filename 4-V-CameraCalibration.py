import numpy as np
import cv2
import glob
import sys

nRows = 6
nCols = 7
dimension = 25

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

objp = np.zeros((nRows*nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

objpoints = []
imgpoints = []


img = cv2.imread('calib.jpg')
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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == True:
        print('Kamera Açıldı')

    else:
        print('HATA!! \nKamera Açılamadı!!')
        exit(1)

    while (True):
        try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_LINEAR)
            newimg = frame

            # cv2.imshow('newimg', newimg)

            h, w = newimg.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            dst = cv2.undistort(newimg, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

            # cv2.imwrite("UndistortedImage.png", dst)
            cv2.imshow('UndistortedImage', dst)
            cv2.imshow('frame', frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                print("Çıkış Yapıldı")
                break
        except:
            print("Beklenmedik Hata!!! ", sys.exc_info()[0])
            raise

cv2.destroyAllWindows()
cap.release()

