import numpy as np
import cv2
import sys

nRows = 6
nCols = 7
dimension = 25

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

objp = np.zeros((nRows*nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(0)
if cap.isOpened() == True:
    print('Kamera Açıldı')

else:
    print('HATA!! \nKamera Açılamadı!!')
    exit(1)


while (True):
    try:
        ret, frame = cap.read()
        if ret != True:
            print('HATA!! Frame Alınamıyor \nYeniden Başlatın'),
            cv2.destroyAllWindows()
            cap.release()
            break
            exit(1)
        frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ret1, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
        if ret1 == True:
            corners1 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (nCols, nRows), corners1, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
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