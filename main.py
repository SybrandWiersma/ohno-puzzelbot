import time

import pyautogui
import cv2
import numpy as np

GREY, BLUE, RED = range(3)
color_grey = (238, 238, 238)
color_blue = (224, 192, 28)
color_red = (74, 56, 255)

samples = np.loadtxt('general_samples.data', np.float32)
responses = np.loadtxt('general_responses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

pyautogui.PAUSE = 0.1

opposite_side = {0: 2, 1: 3, 2: 0, 3: 1}


class Circle:
    def __init__(self, x, y, x_original, y_original, color, number):
        self.x = x
        self.y = y
        self.x_original = x_original
        self.y_original = y_original
        self.color = color
        self.number = number
        self.finished = False


def find_number(img):
    global model
    img = cv2.inRange(img, (200, 200, 100), (255, 255, 255))
    cv2.floodFill(img, None, (0, 0), 0)
    cv2.floodFill(img, None, (0, len(img) - 1), 0)
    cv2.floodFill(img, None, (len(img[0]) - 1, 0), 0)
    cv2.floodFill(img, None, (len(img[0]) - 1, len(img) - 1), 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return -1
    max_size = 0
    max_size_position = 0
    for i, cnt in enumerate(contours):
        if cnt.size > max_size:
            max_size = cnt.size
            max_size_position = i
    x, y, w, h = cv2.boundingRect(contours[max_size_position])
    img = img[y:y + h, x:x + w]
    img_small = cv2.resize(img, (10, 20))
    img_small = img_small.reshape((1, 200))
    img_small = np.float32(img_small)
    ret, results, neighbours, dist = model.findNearest(img_small, 3)

    number = int(results[0][0])
    return number


def get_circles_from_screenshot():
    circles = []
    image = pyautogui.screenshot('screenshot.png')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    circles += find_circles(image, GREY, color_grey)
    circles += find_circles(image, BLUE, color_blue)
    circles += find_circles(image, RED, color_red)
    return circles


def find_circles(image, color, color_value):
    points = []
    img_mask = cv2.inRange(image, color_value, color_value)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt.size < 90:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if color == BLUE:
            img_part = image[y:y + h, x:x + w]
            number = find_number(img_part)
            points.append([int(x + w / 2), int(y + h / 2), color, number])
        else:
            points.append([int(x + w / 2), int(y + h / 2), color])
    return points


def arrange_circles(circles):
    size = int(np.sqrt(len(circles)))
    array = np.empty((size, size), Circle)
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
            array[y][x] = Circle(x, y, circle[0], circle[1], BLUE, circle[3])
        elif circle[2] is GREY:
            array[y][x] = Circle(x, y, circle[0], circle[1], GREY, 0)
        elif circle[2] is RED:
            array[y][x] = Circle(x, y, circle[0], circle[1], RED, -1)
    return array


def circle_connections(circle, circle_array):
    total = [0, 0, 0, 0]
    for i in range(4):
        xy = [0, 0, 0, 0]
        xy[i] = 1
        end_reached = False
        while not end_reached:
            x_check = circle.x + xy[1] - xy[3]
            y_check = circle.y - xy[0] + xy[2]
            if 0 > x_check or x_check >= len(circle_array) \
                    or 0 > y_check or y_check >= len(circle_array[0]):
                end_reached = True
                xy[i] -= 1
                continue
            circle_checking = circle_array[y_check][x_check]
            if circle_checking.color is BLUE:
                xy[i] += 1
                total[i] += 1
            else:
                end_reached = True
    return total


def check_sides(circle, circle_array, connections):
    sides_free = [0, 0, 0, 0]
    for i in range(4):
        xy = [0, 0, 0, 0]
        xy[i] = 1 + connections[i]
        end_reached = False
        blue_row = 0
        while not end_reached:
            x_check = circle.x + xy[1] - xy[3]
            y_check = circle.y - xy[0] + xy[2]
            if 0 > x_check or x_check >= len(circle_array) \
                    or 0 > y_check or y_check >= len(circle_array[0]):
                end_reached = True
                xy[i] -= 1
                continue
            circle_checking = circle_array[y_check][x_check]
            if circle_checking.color is RED:
                end_reached = True
                xy[i] -= 1
                continue
            elif circle_checking.color is BLUE:
                blue_row += 1
                if sum(connections) + (xy[i] - connections[i]) > circle.number:
                    end_reached = True
                    xy[i] -= 1 + blue_row
                    continue

                if circle_checking.number > 0:
                    connections_circle_checking = circle_connections(circle_checking, circle_array)
                    if sum(connections_circle_checking) - connections_circle_checking[opposite_side[i]] + xy[i] \
                            + connections[opposite_side[i]] > circle_checking.number:
                        end_reached = True
                        xy[i] -= 1 + blue_row
                        continue

            xy[i] += 1
        sides_free[i] = xy[i] - connections[i]
    return sides_free


def place_blues(circle, circle_array, connections, sides_free, blues_left):
    blues_final = [0, 0, 0, 0]
    for i in range(4):
        sides_free[i] = (i, sides_free[i])
    sides_free.sort(key=lambda x: x[1])
    for i in range(3, -1, -1):
        total_undecided = 0
        for j in range(i):
            total_undecided += sides_free[j][1]
        if blues_left < total_undecided:
            break
        total_can_place = blues_left - total_undecided

        if total_can_place < sides_free[i][1]:
            blues_final[sides_free[i][0]] = total_can_place
            break
        blues_final[sides_free[i][0]] = sides_free[i][1]
    if sum(blues_final) == 0:
        return
    for i in range(4):
        for j in range(1, blues_final[i] + 1):
            xy = [0, 0, 0, 0]
            xy[i] = connections[i] + j
            x_place = circle.x + xy[1] - xy[3]
            y_place = circle.y - xy[0] + xy[2]
            if circle_array[y_place][x_place].color is BLUE:
                continue
            circle_array[y_place][x_place].color = BLUE
            pyautogui.moveTo(circle_array[y_place][x_place].x_original, circle_array[y_place][x_place].y_original, 0)
            pyautogui.click()
    return


def place_reds_edges(circle, circle_array, connections):
    for i in range(4):
        xy = [0, 0, 0, 0]
        xy[i] = 1 + connections[i]
        x_check = circle.x + xy[1] - xy[3]
        y_check = circle.y - xy[0] + xy[2]
        if 0 > x_check or x_check >= len(circle_array) \
                or 0 > y_check or y_check >= len(circle_array[0]):
            continue
        if circle_array[y_check][x_check].color is RED:
            continue
        circle_array[y_check][x_check].color = RED
        pyautogui.moveTo(circle_array[y_check][x_check].x_original, circle_array[y_check][x_check].y_original, 0)
        pyautogui.click()
        pyautogui.click()

    return


def calculate_answer(circle_array):
    answer_found = False
    while not answer_found:
        for circle_row in circle_array:
            for circle in circle_row:
                if circle.color is BLUE and circle.finished is False and circle.number > 0:
                    connections = circle_connections(circle, circle_array)
                    # check if circle is already finished
                    blues_left = circle.number - sum(connections)
                    if blues_left == 0:
                        circle.finished = True
                        place_reds_edges(circle, circle_array, connections)
                        continue
                    sides_free = check_sides(circle, circle_array, connections)
                    place_blues(circle, circle_array, connections, sides_free, blues_left)
        # check if answer found
        answer_found = True
        for circle_row in circle_array:
            for circle in circle_row:
                if circle.color is BLUE and circle.number > 0 and circle.finished is False:
                    answer_found = False
    # reds on grey's that are not used
    for circle_row in circle_array:
        for circle in circle_row:
            if circle.color is GREY:
                circle.color = RED
                pyautogui.moveTo(circle.x_original, circle.y_original, 0)
                pyautogui.click()
                pyautogui.click()
    return circle_array


def solve_puzzle(circle_array):
    for circle_row in circle_array:
        for circle in circle_row:
            pyautogui.PAUSE = 0
            if circle.color is RED and circle.number == 0:
                pyautogui.moveTo(circle.x_original, circle.y_original, 0)
                pyautogui.click()
                pyautogui.click()
            elif circle.color is BLUE and circle.number == 0:
                pyautogui.moveTo(circle.x_original, circle.y_original, 0)
                pyautogui.click()

    return


def main():
    circles = get_circles_from_screenshot()
    circle_array = arrange_circles(circles)
    calculate_answer(circle_array)
    # solve_puzzle(circle_array)
    print('puzzle solved')


if __name__ == '__main__':
    main()
