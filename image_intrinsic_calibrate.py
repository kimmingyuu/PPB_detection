import enum
from turtle import width
import cv2
import glob
import numpy as np
import time
import math
import matplotlib.pyplot as plt

def get_2dmap(dis):
    list1 = dis
    x = []
    z = []
    for i in list1:
        x.append(i[0])
        z.append(i[1])

    print(x)
    print(z)
    plt.plot(x,z,'o')
    plt.xlabel('x cm')
    plt.ylabel('z cm')
    plt.axis([-50, 50, 0.0 , 100])
    plt.show()

camera_matrix = np.array( [[348.26309945, 0., 332.530534],
                            [0., 347.36040435, 238.04691592], 
                            [0., 0., 1.]])

dist_coeffs = np.array([-0.363409, 0.199566, -4.7e-05, -0.000814, -0.069992])
image_size = (640,480)



# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_32FC1)
# image = cv2.imread("image_undist.png", cv2.IMREAD_COLOR)
# image = cv2.resize(image, (640,480))
# # cv2.imshow("image2", image)
# image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

# # cv2.imwrite("image_undist2.png",image_undist)
# # cv2.imshow("image_undist", image_undist)
# cv2.waitKey(0)

image = cv2.imread("image_undist3.png", cv2.IMREAD_COLOR)
boxes = [[0.514844, 0.705208, 0.045312, 0.068750],
        [0.511719, 0.606250, 0.023438, 0.033333],
        [0.774219, 0.697917, 0.051562, 0.066667],
        [0.255469, 0.712500, 0.048438, 0.066667]]
dis = []

color = [(255,0,0), (0,255,0), (0,0,255), (255,0,255)]
for i, box in enumerate(boxes):    # 아래 위 오른쪽 왼쪽
    center_x, center_y, width, height = box[0], box[1], box[2], box[3] 
    center_x *= 640
    center_y *= 480
    width *= 640/2
    height *= 480/2
    xmin = int(center_x-width)
    ymin = int(center_y-height)
    xmax = int(center_x+width)
    ymax = int(center_y+height)

    # print(xmin, ymin,xmax, ymax)

    CAMERA_HEIGHT = 0.1475
    FOVh = (135.4-42)/2

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color[i], 3)
    cv2.putText(image, f"{i}", (xmin, ymin+25), 1, 2, (0,0,0), 2)
    # cv2.imshow("image_undist", image_undist)
    # cv2.waitKey(0)

    x_center = (xmin+xmax)/2
    y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]
    deltax = (x_center - width/2)
    azimuth = FOVh - (deltax/320*FOVh)

    dz = (1 * CAMERA_HEIGHT / y_norm * 100)
    dx = dz * math.tan(math.pi*(azimuth/180))
    if deltax > 0:
        dx *= -1
    d = dz / math.cos(math.pi * (azimuth/180))
    if azimuth < 0:
        azimuth *= -1

    print("\n\n index : ", i)
    print("azimuth", azimuth)
    print("d : ", d)
    print("dz : ", dz)
    print("dz_alph : ", dz*(1.025)**2)#1.12
    print("dx : ", dx)
    # print("correct_azimuth : ", 26.56 - azimuth)
    # print("correct_d : ", 50.31 - d)
    # print("correct_dz : ", 45 - dz)
    # print("correct_dx : ", 22.5 - dx)

    t = (dx,dz)
    dis.append(t) 


# cv2.imshow("image_undist", image_undist)
# cv2.waitKey(0)

get_2dmap(dis)

cv2.imshow("image_undist3", image)
cv2.waitKey(0)
