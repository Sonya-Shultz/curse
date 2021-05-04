import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


def find_all_contour(img_name):
    white_black(img_name, "res2.jpg", 0.85)
    img = cv2.imread("res2.jpg", cv2.IMREAD_GRAYSCALE)
    blur = cv2.blur(img, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))

    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    canvas = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 100

    for i in range(len(contours)):
        color_contours = (0, 255, 0)
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)

    cv2.imshow('img1', img)
    cv2.waitKey(0)

    cv2.imshow('countur', drawing)
    cv2.waitKey(0)
    return contours, canvas


def rotate_n_perspective_img(rotrect, box, page, img_start):
    (x1, y1), (x2, y2), angle = rotrect
    box1 = [[0, 0], [0, y2], [x2, y2], [x2, 0]]
    box1 = forvard_back(box1)
    box1 = max_x_sort(box1)
    box = np.array(box, np.float32)
    box1 = np.array(box1, np.float32)
    page = aprox_in_array(find_near(page, box), page)
    page = np.array(page, np.float32)
    matrix = cv2.getPerspectiveTransform(page, box1)
    result = cv2.warpPerspective(img_start, matrix, (int(x2), int(y2)))

    cv2.imshow('img transform', result)
    cv2.imwrite("res.jpg", result)
    cv2.waitKey(0)


def how_transform(contours, canvas):
    font = cv2.FONT_HERSHEY_COMPLEX
    page = []
    rotrect = cv2.minAreaRect(contours[0])
    for cnt in contours:
        if cv2.arcLength(cnt, True) > 800:
            approx = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):
                    help_arr = [n[i], n[i + 1]]
                    page.append(help_arr)
                    x = n[i]
                    y = n[i + 1]
                    string = str(x) + " " + str(y)
                    cv2.putText(canvas, string, (x, y), font, 0.5, (0, 255, 0))
                i = i + 1
            rotrect = cv2.minAreaRect(cnt)

    # Зображення отриманого квадрату для тексту
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    cv2.drawContours(canvas, [box], 0, (0, 255, 255), 2)
    cv2.imshow('box', canvas)
    cv2.waitKey(0)
    return rotrect, box, page


def transform_img(img_name):
    img_start = cv2.imread(img_name)
    cv2.imshow('img1', img_start)
    cv2.waitKey(0)

    # Знаходимо контур листка
    contours, canvas = find_all_contour(img_name)

    # Проходимо всі контури та визначаємо матриці трансформації
    rotrect, box, page = how_transform(contours, canvas)

    # матриця переходу і трансформація
    rotate_n_perspective_img(rotrect, box, page, img_start)


def brightness_n_bw_img():
    image = Image.open("res.jpg")

    # Збільшення яскравості на 20%
    new_image = ImageEnhance.Contrast(image).enhance(1.2)
    result = np.array(new_image)
    cv2.imwrite("res.jpg", result)

    # перетворення в чб варіант та його читання
    white_black("res.jpg", "res2.jpg", 0.6)
    result = cv2.imread("res2.jpg")
    cv2.imshow('bw-result', result)
    cv2.waitKey(0)
    return result


def normalize_color_of_img(img_name):
    img = cv2.imread(img_name)
    blur = cv2.blur(img, (40, 40))
    blur = cv2.bitwise_not(blur)
    blur = np.array(blur)*0.6
    img = calc_sum_color_rgb(img, blur, 0.5)
    cv2.imwrite("res.jpg", img)
    cv2.imshow('blur', img)
    cv2.waitKey(0)


def calc_sum_color_rgb(img1, img2, black_range):
    for i in range(len(img1)):
        for j in range(len(img1[i])):
            help_color = np.copy(img1[i][j])
            is_first = True
            if (np.average(img1[i,j]) >= (black_range * 255)):
                for x in range(len(img1[i][j])):
                    help_div = img1[i, j, x] + img2[i, j, x]
                    if help_div > 255:
                        help_div = 255
                    img1[i, j, x] = help_div
            else:
                for x in range(len(img1[i][j])):
                    help_div = img1[i, j, x] + img2[i, j, x] * ((img1[i, j, x])/(black_range * 255))
                    if help_div > 255:
                        help_div = 255
                    img1[i, j, x] = help_div
    return img1


def outline_line_on_img(result):
    # читання в чб форматі і інверсія (для підсилення кольорів)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    cv2.imshow('gray invert', gray)
    cv2.waitKey(0)

    # Пошук та виділення контурів для обведення букв
    edges = cv2.Canny(result, 50, 150, apertureSize=3)
    minLineLength = 1
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength, maxLineGap)
    if lines is not None:
        for l in lines:
            for x1, y1, x2, y2 in l:
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 0), 1)

    cv2.imshow('with countur', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_img(img_name):
    # Вирівнювання кольору працює доволі довго :'(
    normalize_color_of_img(img_name)

    # Стандартизація зображення
    transform_img("res.jpg")

    # це для складних випадків, коли результат не дуже нормальний
    # transform_img("res.jpg")
    # normalize_color_of_img("res.jpg")

    # висвітлення фону паперу, приберання малих косяків
    result = brightness_n_bw_img()

    # підведення ліній
    outline_line_on_img(result)


def find_near(page, box):
    index = []
    for i in range(len(page)):
        lenght = 1000000000
        index.append(0)
        for j in range(len(box)):
            dist = distance_to_point(page[i][0],page[i][1],box[j][0],box[j][1])
            if lenght > dist:
                lenght = dist
                index[i] = j
    return index


def distance_to_point(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def aprox_in_array(index, page):
    max_i = 100000000
    page_retouch = [[max_i, max_i], [max_i, 0], [0, 0], [0, max_i]]
    for i in range(len(index)):
        if index[i] == 0:
            if page[i][0] < page_retouch[0][0]:
                page_retouch[0][0] = page[i][0]
            if page[i][1] < page_retouch[0][1]:
                page_retouch[0][1] = page[i][1]
        if index[i] == 1:
            if page[i][0] < page_retouch[1][0]:
                page_retouch[1][0] = page[i][0]
            if page[i][1] > page_retouch[1][1]:
                page_retouch[1][1] = page[i][1]
        if index[i] == 2:
            if page[i][0] > page_retouch[2][0]:
                page_retouch[2][0] = page[i][0]
            if page[i][1] > page_retouch[2][1]:
                page_retouch[2][1] = page[i][1]
        if index[i] == 3:
            if page[i][0] > page_retouch[3][0]:
                page_retouch[3][0] = page[i][0]
            if page[i][1] < page_retouch[3][1]:
                page_retouch[3][1] = page[i][1]
    return page_retouch


def forvard_back(arr):
    arr2 = []
    for i in range(len(arr)):
        arr2.append(arr[len(arr)-i-1])
    return arr2


def max_x_sort(arr):
    arr2 = []
    max_s = 10000000000000000
    max_ind = 0
    for i in range(len(arr)):
        if max_s > arr[i][0]:
            max_s = arr[i][0]
            max_ind = i
    for el in arr[max_ind:4]:
        arr2.append(el)
    for el in arr[0:max_ind]:
        arr2.append(el)
    return arr2


def white_black(source_name, result_name, brightness):
    source = Image.open(source_name)
    result = Image.new('RGB', source.size)
    separator = 255 / brightness / 2 * 3
    for x in range(source.size[0]):
        for y in range(source.size[1]):
            r, g, b = source.getpixel((x, y))
            total = r + g + b
            if total > separator:
                result.putpixel((x, y), (255, 255, 255))
            else:
                result.putpixel((x, y), (0, 0, 0))
    result.save(result_name, "JPEG")


if __name__ == '__main__':
    test_img("test4.jpg")
