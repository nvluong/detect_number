import cv2
import imutils
from skimage import measure
import numpy as np
import os
from data_util import *

def segmentation(image, candidates):
    cv2.imshow('origin', image)
    cv2.waitKey(0)
    V = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #-------------------------
    # xóa viền dọc
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    # xóa viền ngang
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 33))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # remove_vertical =cv2.dilate(remove_vertical , vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 5)

    V = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(V, 200, 255)
    cv2.imshow('hehe1', mask)
    cv2.waitKey(0)

    mask = cv2.bitwise_not(mask)
    cv2.imshow('hehe', mask)
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    labels = measure.label(mask, connectivity=2, background=0)

    print("type of labels ", type(labels))
    print(labels[0].shape)

    dir_path = r'./data_new'
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print('File count:', count)

    for label in np.unique(labels):

        # if this is background label, ignore it
        if label == 0:
            continue

        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255



        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)
            # rule to determine characters
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(image.shape[0])

            print("aspectRatio",aspectRatio)
            print("solidity", solidity)
            print("heightRatio", heightRatio)

            print('h = ', h)
            print('w = ', w)

            if 0.14 < aspectRatio < 2.3 and 0.12 < solidity < 0.91 and 0.13 < heightRatio < 1 and 16 < h < 96 and 8 < w < 122:
                count +=1

                print(mask[y:y + h, x:x + w].shape)

                candidate = np.array(mask[y:y + h, x:x + w])

                square_candidate = candidate

                square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)


                square_candidate = square_candidate.reshape((28, 28, 1))

                candidates.append((square_candidate, (y, x)))
                print('type of candidates', type(candidates))
                print('len ', len(candidates))

def recognizeChar(candidates, candidates1, dict_temp, model):
    characters = []
    coordinates = []
    if len(candidates) == 0:
        candidates1.append(None)
        return 0
    for char, coordinate in candidates:
        characters.append(char)
        coordinates.append(coordinate)

    characters = np.array(characters)
    result = model.predict(characters)
    print('rs', result)
    result_idx = np.argmax(result, axis=1)
    print('rs', result_idx[0])
    str = ""
    for i in range(len(result_idx)):
        candidates1.append((dict_temp[result_idx[i]], coordinates[i]))
        str+= dict_temp[result_idx[i]]
    print('str ', str)

def predict(image, dict_temp, model):
    image = image
    dem = 0
    candidates = []
    candidates1 = []
    LpRegion = image

    cv2.imshow('LPRegion', LpRegion)

    cv2.waitKey(0)
    # segmentation
    segmentation(LpRegion, candidates)
    recognizeChar(candidates, candidates1, dict_temp, model)

    license_plate = format(candidates1)

     # draw labels
    image = draw_labels_and_boxes(image, license_plate)
    print("dem = ", dem)
    return image





