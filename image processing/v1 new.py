import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from time import sleep
import random
import math

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import datetime
import os


def capture_image():
    # capture image for processing
    # cam = cv2.VideoCapture(0)
    # ret, frame = cam.read()    
    # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    # # cv2.imread(frame)
    # img = frame[100:590, 90:425]
    img = cv2.imread('IMG_5563_med.jpg')

    return img


def blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray ", gray)
    gray[np.where(gray <= [190])] = [0]
    kernel = np.ones((5, 5), np.uint8)
    denoised_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return denoised_img


def find_contours(img, blank_image):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # print(len(contours))
    shapes_of_rice = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:
            shapes_of_rice.append(contour)

    # print(len(shapes_of_rice))

    for shape in shapes_of_rice:
        cv2.drawContours(blank_image, shape, -1, random_color(), 2)
        points_4cut = find_centroid(shape, img, blank_image)
        find_angle(points_4cut, blank_image)
    # cv2.imshow("shape ", cv2.resize(blank_image, (0,0), fx=0.5, fy=0.5))
    return 0


def find_centroid(shape, img, blank_image):
    points_4cut_list = []

    for n in range(len(shape)):
        path = []
        points = []
        for i in range(n - 8, n + 9):
            if i >= len(shape):
                x = i - len(shape) + 1
                path.append(shape[x][0])
            else:
                path.append(shape[i][0])

        point_1 = ((path[0][0] + path[1][0] + path[2][0] + path[3][0]) / 4,
                   (path[0][1] + path[1][1] + path[2][1] + path[3][1]) / 4)
        point_2 = path[4]
        point_3 = ((path[5][0] + path[6][0] + path[7][0] + path[8][0]) / 4,
                   (path[5][1] + path[6][1] + path[7][1] + path[8][1]) / 4)

        centroid = (((point_1[0] + point_2[0] + point_3[0]) / 3),
                    ((point_1[1] + point_2[1] + point_3[1]) / 3))

        length_ct2t = ((centroid[0] - point_2[0]) ** 2 + (centroid[1] - point_2[1]) ** 2) ** 0.5
        if length_ct2t > 0.5:
            if img[int(round(centroid[1])), int(round(centroid[0]))] == 0:
                if len(points_4cut_list) == 0:
                    points.append((path[0][0], path[0][1]))
                    points.append((point_2[0], point_2[1]))
                    points.append((path[8][0], path[8][1]))
                    points.append(centroid)
                    points_4cut_list.append(points)
                    for i, point in enumerate(points):
                        if i == 0:
                            blank_image[point[1], point[0]] = (120, 120, 0)
                        elif i == 1:
                            blank_image[point[1], point[0]] = (127, 0, 255)
                        elif i == 2:
                            blank_image[point[1], point[0]] = (255, 255, 255)
                        else:
                            blank_image[int(round(point[1])), int(round(point[0]))] = (0, 0, 255)

                else:
                    count = 0
                    for i, cent in enumerate(points_4cut_list):
                        if abs(cent[3][0] - round(centroid[0])) <= 1.5 and abs(cent[3][1] - round(centroid[1])) <= 1.5:
                            length_old_cent = ((points_4cut_list[i][2][0] - points_4cut_list[i][0][0]) ** 2 + (
                                        points_4cut_list[i][2][1] - points_4cut_list[i][0][1]) ** 2) ** 0.5
                            length_new_cent = ((path[8][0] - path[0][0]) ** 2 + (path[8][1] - path[0][1]) ** 2) ** 0.5
                            if length_new_cent < length_old_cent:
                                points_4cut_list[i][0] = (path[0][0], path[0][1])
                                points_4cut_list[i][1] = (path[4][0], path[4][1])
                                points_4cut_list[i][2] = (path[8][0], path[8][1])
                                points_4cut_list[i][3] = (centroid[0], centroid[1])

                        else:
                            count += 1
                    if count == len(points_4cut_list):
                        points.append((path[0][0], path[0][1]))
                        points.append((point_2[0], point_2[1]))
                        points.append((path[8][0], path[8][1]))
                        points.append(centroid)
                        points_4cut_list.append(points)
                        for i, point in enumerate(points):
                            if i == 0:
                                blank_image[point[1], point[0]] = (120, 120, 0)
                            elif i == 1:
                                blank_image[point[1], point[0]] = (127, 0, 255)
                            elif i == 2:
                                blank_image[point[1], point[0]] = (255, 255, 255)
                            else:
                                blank_image[int(round(point[1])), int(round(point[0]))] = (0, 0, 255)

    return points_4cut_list


def find_angle(points, image):
    points_copy = points
    i = 0
    line_4cut = []

    while i < len(points_copy):
        q = 0
        for j in range(i + 1, len(points_copy)):
            angle_a = getAngle(points_copy[i][0], points_copy[i][3], points_copy[j][3], 1)
            angle_b = getAngle(points_copy[i][2], points_copy[i][3], points_copy[j][3], 2)
            angle_c = getAngle(points_copy[j][0], points_copy[j][3], points_copy[i][3], 1)
            angle_d = getAngle(points_copy[j][2], points_copy[j][3], points_copy[i][3], 2)

            if abs(angle_a) <= 180 and abs(angle_b) <= 180 and abs(angle_c) <= 180 and abs(angle_d) <= 180:
                length_line = ((points_copy[j][1][0] - points_copy[i][1][0]) ** 2 + (
                            points_copy[j][1][1] - points_copy[i][1][1]) ** 2) ** 0.5
                line_4cut.append([points_copy[i][1], points_copy[j][1], length_line])

        i = i + 1

    line_4cut.sort(key=takeSecond)

    i = 0
    while i < len(line_4cut):
        cv2.line(image, line_4cut[i][0], line_4cut[i][1], (255, 255, 0), 1)
        a = i + 1
        while a < len(line_4cut):
            if line_4cut[i][0] == line_4cut[a][0]:
                line_4cut.pop(a)
                a = a - 1
            elif line_4cut[i][1] == line_4cut[a][1]:
                line_4cut.pop(a)
                a = a - 1

            a = a + 1
        i = i + 1
    return


def takeSecond(elem):
    return elem[2]


def getAngle(a, b, c, num):
    slope_x = c[0] - b[0]
    slope_y = c[1] - b[1]
    if slope_x == 0:
        m = 1000000
    else:
        m = (slope_y) / (slope_x)
        if m == 0:
            m = 0.000001

    x = ((1 / m) * (a[1] - b[1])) + b[0]
    y = (m * (a[0] - b[0])) + b[1]

    C = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    A = ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2) ** 0.5
    B = ((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2) ** 0.5
    arccos = (A * A + C * C - B * B) / (2.0 * A * C)

    if arccos > 1:
        arccos = 1
    elif arccos < -1:
        arccos = -1

    angle = np.degrees(np.arccos(arccos))
    if b[0] <= c[0] and b[1] <= c[1]:
        if num == 1:
            if a[0] >= x and a[1] <= y:
                result = angle
            else:
                result = 360 - angle
        else:
            if a[0] <= x and a[1] >= y:
                result = angle
            else:
                result = 360 - angle

    elif b[0] >= c[0] and b[1] >= c[1]:
        if num == 1:
            if a[0] <= x and a[1] >= y:
                result = angle
            else:
                result = 360 - angle
        else:
            if a[0] >= x and a[1] <= y:
                result = angle
            else:
                result = 360 - angle

    elif b[0] <= c[0] and b[1] >= c[1]:
        if num == 1:
            if a[0] <= x and a[1] <= y:
                result = angle
            else:
                result = 360 - angle
        else:
            if a[0] >= x and a[1] >= y:
                result = angle
            else:
                result = 360 - angle

    elif b[0] >= c[0] and b[1] <= c[1]:
        if num == 1:
            if a[0] >= x and a[1] >= y:
                result = angle
            else:
                result = 360 - angle
        else:
            if a[0] <= x and a[1] <= y:
                result = angle
            else:
                result = 360 - angle

    return result


def random_color():
    color = (random.randint(0, 255), random.randint(0, 255),
             random.randint(0, 255))
    return color


def find_numNlength_of_rice(img, ratio):
    data2_save = []
    class_1 = 0
    class_2 = 0
    class_3 = 0
    short = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours_external, _1 = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)

    for cont_ex in contours_external:
        i = 0
        for cont in contours:
            comparison = np.array_equal(cont, cont_ex)
            if comparison == 1:
                i = list[i]
                contours.pop(i)
            i += 1

    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)
        # print(area)
        if area < 50:
            contours.pop(i)

    num_of_rice = (len(contours))
    i = 0
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        height = rect[1][0] * ratio
        wide = rect[1][1] * ratio

        if wide > height:
            temp = wide
            wide = height
            height = temp

        if height >= 7:
            class_1 += 1

        elif height >= 6.6:
            class_2 += 1

        elif height >= 6.2:
            class_3 += 1

        elif height < 6.2:
            short += 1
        else:
            print('sus')
        i += 1
    data4display = [('TORAL ' + str(num_of_rice)),
                    ('CLS1 ' + str(class_1) + ' : CLS2 ' + str(class_2)),
                    ('CLS3 ' + str(class_3) + ' : SHORT ' + str(short))]
    data2_save.append(num_of_rice)
    data2_save.append(class_1)
    data2_save.append(class_2)
    data2_save.append(class_3)
    data2_save.append(short)

    return data2_save


def get_ratio_pixel2mm(img):
    median = cv2.medianBlur(img, 11)
    dst = cv2.fastNlMeansDenoisingColored(median, None, 10, 10, 7, 21)
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    lower_color = np.array([67, 69, 106])
    uper_color = np.array([94, 255, 255])
    mask = cv2.inRange(hsv, lower_color, uper_color)
    # cv2.imshow("aa", cv2.resize(mask, (0,0), fx=0.5, fy=0.5))
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow("bb", cv2.resize(mask_inv, (0,0), fx=0.5, fy=0.5))
    result = cv2.bitwise_and(img, img, mask=mask_inv)
    # cv2.imshow("11", cv2.resize(result, (0,0), fx=0.5, fy=0.5))
    # cv2.imshow("mask", mask)
    # cv2.imshow("result1", cv2.resize(result, (0,0), fx=0.5, fy=0.5))

    # mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    total_length = 0
    area_of_ref = []
    print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            area_of_ref.append(area)

    print(len(area_of_ref))
    for area in area_of_ref:
        length = area ** 0.5
        print("length", length)
        total_length += length

    final_length = 10 / 1
    print("final_length", final_length)
    ratio = 20 / final_length
    print("ratio", ratio)
    print("final_length", final_length * ratio)
    return ratio




def save_data(data, img):
    # date
    now = datetime.datetime.now()
    year = "{:02d}".format(now.year)
    month = "{:02d}".format(now.month)
    day = "{:02d}".format(now.day)
    day_month_year = "{}-{}-{}".format(day, month, year)

    dirname = "/Users/austinc/Desktop/image process/" + day_month_year

    try:
        os.mkdir(dirname)
        print('created')
    except FileExistsError:
        print('already')

    content = []
    order_of_pic = 1
    try:
        file = open(dirname + "/" + day_month_year + ".txt", "r")
        for line in file:
            content_each_line = line.split(" ")
            content.append(content_each_line)
            order_of_pic = int(content[len(content) - 1][0]) + 1
    except FileNotFoundError:
        file = open(dirname + "/" + day_month_year + ".txt", "a")
        order_of_pic = 1
    # print(order_of_pic)

    file = open(dirname + "/" + day_month_year + ".txt", "a+")
    file.write(str(order_of_pic) + " " + str(data[0]) + " " + str(data[1]) + " " + str(data[2]) +
               " " + str(data[3]) + " " + str(data[4]) + " ")
    file.write("\n")
    file.close()

    cv2.imwrite(dirname + "/" + str(order_of_pic) + ".png", img)

    return 0


while True:
    start_time = time.time()
    # Loading Images
    img = capture_image()
    img = cv2.resize(img, (900, 1200), fx=3, fy=3)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.imshow("Original ", cv2.resize(img, (900, 1200), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    ratio = get_ratio_pixel2mm(img)
    print(ratio)
    cv2.imshow("blank ", blank_image)
    cv2.waitKey(0)
    denoised_image = blur(img)
    find_contours(denoised_image, blank_image)
    # data2_save = find_numNlength_of_rice(blank_image, ratio)
    # print(data2_save)
    cv2.imshow("Bounding", cv2.resize(blank_image, (900, 1200), fx=0.5, fy=0.5))
    cv2.waitKey(0)

    # save_data(data2_save, img)
    end_time = time.time()
    print(end_time - start_time)
    break
