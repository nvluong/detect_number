import numpy as np
import cv2

def order_points(coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect
def format(candidates):
    first_line = []
    second_line = []

    if candidates == None:
        return 0
    print('shape candidate ', len(candidates))
    print(candidates)
    for candidate, coordinate in candidates:
        if candidates[0][1][0] + 40 > coordinate[0]:
            first_line.append((candidate, coordinate[1]))
        else:
            second_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, key=take_second)
    second_line = sorted(second_line, key=take_second)

    if len(second_line) == 0:  # if license plate has 1 line
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:  # if license plate has 2 lines
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join(
            [str(ele[0]) for ele in second_line])
    # print('license ', license_plate)
    return license_plate

def draw_labels_and_boxes(image, labels):
    x_min = 28
    y_min = 28
    if labels == 0:
        image = cv2.putText(image, "", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255),
                            thickness=2)
    else:
        image = cv2.putText(image, labels, (x_min , y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255),thickness=2)

    return image