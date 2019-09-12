import numpy as np
import cv2
import yaml

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

blobParams = cv2.SimpleBlobDetector_Params()

blobParams.minThreshold = 8
blobParams.maxThreshold = 255

blobParams.filterByArea = True
blobParams.minArea = 64
blobParams.maxArea = 2500

blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

blobDetector = cv2.SimpleBlobDetector_create(blobParams)

objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)

objpoints = []
imgpoints = []


cap = cv2.VideoCapture(0)
if cap.isOpened() == True:
    print('Kamera Açıldı')
else:
    print('HATA!! \nKamera Açılamadı!!')
    exit(1)

found = 0
while(found < 10):
# while(1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        found += 1
    cv2.putText(im_with_keypoints, "Found Circle Grid Count: {}".format(found), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


    cv2.imshow("img", im_with_keypoints)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print("Çıkış Yapıldı")
        break


cap.release()
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
# data = {'camera_matrix': mtx, 'dist_coeff': dist}

with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)
print('Kalibrasyon Tamamlandı')

# with open('calibration.yaml') as f:
#     loadeddict = yaml.load(f)
# mtxloaded = loadeddict.get('camera_matrix')
# distloaded = loadeddict.get('dist_coeff')

cv_file = cv2.FileStorage("test.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)
cv_file.release()

