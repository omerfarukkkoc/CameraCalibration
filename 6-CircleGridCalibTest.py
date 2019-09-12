

import numpy as np
import cv2
import yaml
import numpy as np
import glob
import sys

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
mtxloaded = loadeddict.get('camera_matrix')
distloaded = loadeddict.get('dist_coeff')

# aa = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
# matrixx = aa.getNode("camera_matrix").mat()
#
#
# yaml_data = np.asarray(cv2.Load("calibration.yaml"))
# mtxx = yaml_data.getfield('camera_matrix')
# dstt = yaml_data.getfield('dist_coeff')

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
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtxloaded, distloaded, (w, h), 1, (w, h))

        dst = cv2.undistort(newimg, newcameramtx, distloaded, None, newcameramtx)

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