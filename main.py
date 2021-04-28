import cv2
import numpy as np
import sys


def print_hi(name):
    print(f'Hi, {name}')


def test_img(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('text', img)
    # cv2.waitKey(0)
    # cv2.imwrite('greyTest1.jpg', img)
    blur = cv2.blur(img, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    canvas = np.ones((img.shape[0], img.shape[1], 3), np.uint8)*100
    for i in range(len(contours)):
        color_contours = (0, 255, 0)
        #color = (255, 0, 0)
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        #cv2.drawContours(canvas, hull, i, color, 1, 8)
    cv2.imshow('text', img)
    cv2.waitKey(0)
    cv2.imshow('text', drawing)
    cv2.waitKey(0)

    # Проходя через все контуры, найденные на изображении.
    font = cv2.FONT_HERSHEY_COMPLEX
    page = [] # ([[0, 0], [0, 0], [0, 0], [0, 0]])
    rotrect = cv2.minAreaRect(contours[0])
    for cnt in contours:
        if cv2.arcLength(cnt, True) > 500:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):
                    help_arr = [n[i], n[i+1]]
                    page.append(help_arr)
                    x = n[i]
                    y = n[i + 1]
                    string = str(x) + " " + str(y)
                    cv2.putText(canvas, string, (x, y), font, 0.5, (0, 255, 0))
                i = i + 1
            rotrect = cv2.minAreaRect(cnt)
    print(page)
    # РЕДАГУВАННЯ МАСИВУ ДЛЯ ВІДПОВІДНОСТІ
    # a = page[1]
    # page[1] = page[0]
    # page[0] = a
    # a = page[2]
    # page[2] = page[3]
    # page[3] = a

    # коробочка по фрейду
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    print(box)
    cv2.drawContours(canvas, [box], 0, (0, 255, 255), 2)
    cv2.imshow('text', canvas)
    cv2.waitKey(0)

    # матриця переходу і трансформація
    page = np.array(page, np.float32)
    (x1, y1), (x2, y2), angle = rotrect
    #x = int(np.sqrt((box[])))
    #y = int(np.sqrt())
    box1 = [[0, 0], [0, y2], [x2, y2], [x2, 0]]
    print("AAAAAAAA", box1)
    print("AAAAAAAA", page)
    box = np.array(box, np.float32)
    box1 = np.array(box1, np.float32)
    matrix = cv2.getPerspectiveTransform(page, box1)
    h, w = img.shape
    result = cv2.warpPerspective(img, matrix, (int (x2), int(y2)))# (int(box[1][0]-box[0][0]), int(box[2][1]-box[0][1])))
    print(matrix)


    # Wrap the transformed image
    cv2.imshow('frame1', result)  # Transformed Capture
    cv2.waitKey(0)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    test_img("test4.jpg")
