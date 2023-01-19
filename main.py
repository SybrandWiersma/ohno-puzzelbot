import pyautogui
import cv2
import numpy as np

GREY, BLUE, RED = range(3)
color_grey = (238, 238, 238)
color_blue = (224, 192, 28)
color_red = (74, 56, 255)

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size,1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)


# class circle():
#     def __init__(self, color):


def find_number(img):
    global model
    img = cv2.inRange(img, (200, 200, 100), (255, 255, 255))
    cv2.floodFill(img, None, (0, 0), 0)
    cv2.floodFill(img, None, (0, len(img) - 1), 0)
    cv2.floodFill(img, None, (len(img[0]) - 1, 0), 0)
    cv2.floodFill(img, None, (len(img[0]) - 1, len(img) - 1), 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_size = 0
    max_size_position = 0
    for i, cnt in enumerate(contours):
        if cnt.size > max_size:
            max_size = cnt.size
            max_size_position = i
    x, y, w, h = cv2.boundingRect(contours[max_size_position])
    img = img[y:y+h, x:x+w]
    img_small = cv2.resize(img, (10, 20))
    img_small = img_small.reshape((1, 200))
    img_small = np.float32(img_small)
    ret, results, neighbours, dist = model.findNearest(img_small, 3)

    number = int(results[0][0])
    return number


def find_circles(image, color, color_value):
    points = []
    img_mask = cv2.inRange(image, color_value, color_value)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt.size < 90:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if color == BLUE:
            img_part = image[y:y+h, x:x+w]
            number = find_number(img_part)
            points.append([int(x + w/2), int(y+h/2), color, number])
        else:
            points.append([int(x + w/2), int(y+h/2), color])
    return points


def arrange_circles(circles):
    size = int(np.sqrt(len(circles)))
    array = np.empty((size, size), int)
    position_dict = [dict(), dict()]

    for i in range(2):
        position_set = set()
        for circle in circles:
            position_set.add(circle[i])
        position_set = sorted(position_set)
        last_value = None
        j = 0
        for value in position_set:
            if last_value is not None and 10 < value - last_value > -10:
                j += 1
            last_value = value
            position_dict[i][value] = j

    for circle in circles:
        x = position_dict[0][circle[0]]
        y = position_dict[1][circle[1]]
        if circle[2] is BLUE:
            array[y][x] = circle[3]
        elif circle[2] is GREY:
            array[y][x] = 0
        elif circle[2] is RED:
            array[y][x] = -1
    return array


def main():
    circles = []
    image = pyautogui.screenshot('screenshot.png')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    circles += find_circles(image, GREY, color_grey)
    circles += find_circles(image, BLUE, color_blue)
    circles += find_circles(image, RED, color_red)
    circle_array = arrange_circles(circles)

    print('test')


if __name__ == '__main__':
    main()
