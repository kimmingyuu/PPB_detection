import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def get_2dmap(dis): # plt 그리기
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

def background(): # 격자 배경 만들기
    color = [(255,0,0), (0,255,0), (0,0,255), (255,0,255)]
    mask = np.zeros((320,320), np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    color = (100,100,100)

    cv2.line(mask, (160, 0), (160,320), color, 1)
    cv2.line(mask, (0, 160), (320,160), color, 1)

    cv2.line(mask, (0, 40), (320,40), color, 1)
    cv2.line(mask, (0, 80), (320,80), color, 1)
    cv2.line(mask, (0, 120), (320,120), color, 1)
    cv2.line(mask, (0, 160), (320,160), color, 1)
    cv2.line(mask, (0, 200), (320,200), color, 1)
    cv2.line(mask, (0, 240), (320,240), color, 1)
    cv2.line(mask, (0, 280), (320,280), color, 1)

    cv2.line(mask, (40, 0), (40,320), color, 1)
    cv2.line(mask, (80, 0), (80,320), color, 1)
    cv2.line(mask, (120, 0), (120,320), color, 1)
    cv2.line(mask, (160, 0), (160,320), color, 1)
    cv2.line(mask, (200, 0), (200,320), color, 1)
    cv2.line(mask, (240, 0), (240,320), color, 1)
    cv2.line(mask, (280, 0), (280,320), color, 1)
    return mask

def calibration(): # calibration && save
    image_size = (640,480)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_32FC1)
    image = cv2.imread("image_undist.png", cv2.IMREAD_COLOR)
    image = cv2.resize(image, image_size)
    # cv2.imshow("image2", image)
    image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # cv2.imwrite("image_undist2.png",image_undist)
    # cv2.imshow("image_undist", image_undist)
    cv2.waitKey(0)

camera_matrix = np.array( [[348.26309945, 0., 332.530534],
                            [0., 347.36040435, 238.04691592], 
                            [0., 0., 1.]])
dist_coeffs = np.array([-0.363409, 0.199566, -4.7e-05, -0.000814, -0.069992])

image = cv2.imread("image_undist3.png", cv2.IMREAD_COLOR)
boxes = [[0.514844, 0.705208, 0.045312, 0.068750],
        [0.511719, 0.606250, 0.023438, 0.033333],
        [0.774219, 0.697917, 0.051562, 0.066667],
        [0.255469, 0.712500, 0.048438, 0.066667]]
dis = []

for i, box in enumerate(boxes): 
    center_x, center_y, width, height = box[0], box[1], box[2], box[3] 
    center_x *= 640
    center_y *= 480
    width *= 640/2
    height *= 480/2
    # xmin = int(box[0])
    # ymin = int(box[1])
    # width = int(box[2])
    # height = int(box[3])
    # xmax = int(xmin+width)
    # ymax = int(ymin+height)

    xmin = int(center_x-width)
    ymin = int(center_y-height)
    xmax = int(center_x+width)
    ymax = int(center_y+height)

    # print(xmin, ymin,xmax, ymax)

    CAMERA_HEIGHT = 0.1475
    FOVh = (135.4-42)/2

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
    cv2.putText(image, f"{i}", (xmin, ymin), 1, 1, (0,0,0), 2)
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

    # print("\n\n index : ", i)
    # print("azimuth", azimuth)
    # print("d : ", d)
    # print("dz : ", dz)
    # print("dz_alph : ", dz*(1.025)**2)#1.12
    # print("dx : ", dx)

    # print("correct_azimuth : ", 26.56 - azimuth)
    # print("correct_d : ", 50.31 - d)
    # print("correct_dz : ", 45 - dz)
    # print("correct_dx : ", 22.5 - dx)

    t = (dx,dz)
    dis.append(t) 


# cv2.imshow("image_undist", image_undist)
# cv2.waitKey(0)

# get_2dmap(dis)

# mask = background()
mask = cv2.imread("mask.png")

cv2.imwrite("mask.png", mask)

for list in dis:
    x = round(list[0]) + 160
    # x += abs(160-x)*3
    if x >= 160:
        x += -(160-x) * 3
    else:
        x += (x-160) * 3
    # y = 320 - round(list[1])
    # y -= (320 - (320 - round(list[1])))
    y = 320 - round(list[1]) * 2

    cv2.circle(mask, (int(x),int(y)), 10, (255, 255, 255), -1)
    cv2.putText(mask, f"({int(list[0])}, {int(list[1])})", (int(x - 20),int(y + 30)), 1, 1, (255, 255, 255), 1)

cv2.imshow("mask",mask)
cv2.imshow("image_undist3", image)
cv2.waitKey(0)
