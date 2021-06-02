import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import math


class RetouchNSplitImg:
    def __init__(self, img_name):
        self.img_name = img_name

    def sort_cnts(self, save_prefix, cnts, image):
        sorted_cnts = []
        if save_prefix == "r":
            sorted_cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
        else:
            if save_prefix == "w" or save_prefix == "s":
                sorted_cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
                ind = 0
                help_arr = []
                coef = 1.8
                if save_prefix == "s":
                    coef = 1.5
                for el in sorted_cnts:
                    if cv2.boundingRect(el)[1] > (np.size(image, 0)) / coef:
                        help_arr.append(ind)
                    ind += 1
                for i in range(len(help_arr)):
                    help_el = sorted_cnts.pop(help_arr[i] - i)
                    sorted_cnts.append(help_el)
            else:
                return cnts
        return sorted_cnts

    def divide_to(self, to_height, to_side, parent_name, save_prefix, is_symbol):
        image = Image.open(parent_name)
        image = np.array(image)
        parent_name = "img/.jpg" if parent_name == 'img/res2.jpg' else parent_name
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # розділяємо на контури
        kernel = np.ones((to_height, to_side), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        # cv2.imshow('dilated', img_dilation)
        # cv2.waitKey(0)

        # знайти і відсортувати контури
        cnts, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_cnts = self.sort_cnts(save_prefix, cnts, image)
        row_counter = 0
        for i, ctr in enumerate(sorted_cnts):
            x, y, w, h = cv2.boundingRect(ctr)
            if (w > 10 and 10 < h) or is_symbol:
                if save_prefix == "s" and 2.6 * h < w:
                    part = int(w / h)
                    for a in range(part):
                        roi = image[y:y + h, x + math.ceil((w / part) * a):x + math.ceil((w / part) * (a + 1))]
                        roi = cv2.resize(roi, (28, 28))
                        if self.is_trash(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)):
                            name = str(parent_name.rpartition('.')[0]) + str(save_prefix) + '{}.jpg'.format(row_counter)
                            cv2.imwrite(name, roi)
                            row_counter += 1
                else:
                    roi = image[y:y + h, x:x + w]
                    if h > 0.05 * len(image) and w > 0.05 * len(image):
                        if is_symbol:
                            roi = cv2.resize(roi, (28, 28))
                        if save_prefix in "r":
                            roi = cv2.resize(roi, (int(w*(400/h)), 400))
                        name = str(parent_name.rpartition('.')[0]) + str(save_prefix) + '{}.jpg'.format(row_counter)
                        if save_prefix == "t":
                            name = str(parent_name.rpartition('.')[0]) + '.jpg'
                            im = Image.fromarray(np.uint8(roi))
                            im.save(name, "JPEG")
                            row_counter += 1
                        else:
                            if self.is_trash(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)):
                                cv2.imwrite(name, roi)
                                row_counter += 1
        return row_counter

    def find_all_contour(self, img_name):
        self.white_black(img_name, "img/res2.jpg", 0.85)
        img = Image.open("img/res2.jpg")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(img, (3, 3))
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = []

        # рахуємо довжину кожного контуру
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))

        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        canvas = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 100

        for i in range(len(contours)):
            color_contours = (0, 255, 0)
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)

        #cv2.imshow('img1', img)
        #cv2.waitKey(0)

       # cv2.imshow('countur', drawing)
       # cv2.waitKey(0)
        return contours, canvas

    def rotate_n_perspective_img(self, rotrect, box, page, img_start):
        (x1, y1), (x2, y2), angle = rotrect
        if angle > 45:
            (x2, y2) = (y2, x2)
        box1 = [[0, 0], [0, y2], [x2, y2], [x2, 0]]
        box1 = self.forvard_back(box1)
        box = np.array(box, np.float32)
        box1 = np.array(box1, np.float32)
        page = self.aprox_in_array(self.find_near(page,  box), page)
        page =self.rotate_arr_page(page, [x2, 0])
        page = np.array(page, np.float32)
        matrix = cv2.getPerspectiveTransform(page, box1)
        result = cv2.warpPerspective(img_start, matrix, (int(x2), int(y2)))

        #cv2.imshow('img transform', result)
        cv2.imwrite("img/res.jpg", result)
        #cv2.waitKey(0)

    def rotate_arr_page(self, page, corner):
        min_len = 100000000
        index = 0
        for i in range(len(page)):
            if min_len > self.distance_to_point(page[i][0], page[i][1], corner[0], corner[1]):
                min_len = self.distance_to_point(page[i][0], page[i][1], corner[0], corner[1])
                index = i
        arr2 = []
        for el in page[index:4]:
            arr2.append(el)
        for el in page[0:index]:
            arr2.append(el)
        return arr2

    def how_transform(self, contours, canvas):
        font = cv2.FONT_HERSHEY_COMPLEX
        page = []
        rotrect = cv2.minAreaRect(contours[0])
        for cnt in contours:
            if cv2.arcLength(cnt, True) > 1000:
                approx = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True)
                cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 5)
                n = approx.ravel()
                i = 0
                for j in n:
                    if i % 2 == 0:
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
        #cv2.imshow('box', canvas)
        #cv2.waitKey(0)
        return rotrect, box, page

    def transform_img(self, img_name):
        img_start = Image.open(img_name)
        img_start = np.array(img_start)
       # cv2.imshow('img1', img_start)
        #cv2.waitKey(0)

        # Знаходимо контур листка
        contours, canvas = self.find_all_contour(img_name)

        # Проходимо всі контури та визначаємо матриці трансформації
        rotrect, box, page = self.how_transform(contours, canvas)

        # матриця переходу і трансформація
        self.rotate_n_perspective_img(rotrect, box, page, img_start)

    def brightness_n_bw_img(self):
        image = Image.open("img/res.jpg")

        # Збільшення яскравості на 20%
        new_image = ImageEnhance.Contrast(image).enhance(1.0)
        result = np.array(new_image)
        cv2.imwrite("img/res.jpg", result)

        # перетворення в чб варіант та його читання
        self.white_black("img/res.jpg", "img/res2.jpg", 0.65)
        result = Image.open("img/res2.jpg")
        result = np.array(result)
        #cv2.imshow('bw-result', result)
        #cv2.waitKey(0)
        return result

    def normalize_color_of_img(self, img_name):
        img = Image.open(img_name)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(img, (40, 40))
        blur = cv2.bitwise_not(blur)
        blur = np.array(blur) * 0.8
        img = self.calc_sum_color_rgb(img, blur, 0.6)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("img/res.jpg", img)
       # cv2.imshow('blur', img)
       # cv2.waitKey(0)

    def calc_sum_color_rgb(self, img1, img2, black_range):
        for i in range(len(img1)):
            for j in range(len(img1[i])):
                help_color = np.copy(img1[i][j])
                if img1[i][j] >= (black_range * 255):
                    help_div = img1[i, j] + img2[i, j]
                    if help_div > 255:
                        help_div = 255
                    img1[i, j] = help_div
                else:
                    help_div = img1[i, j] + img2[i, j] * ((img1[i, j]) / (black_range * 255))
                    if help_div > 255:
                        help_div = 255
                    img1[i, j] = help_div
        return img1

    def outline_line_on_img(self, result, name):
        # читання в чб форматі і інверсія (для підсилення кольорів)
        result = 255-result
        ret, thresh = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((1, 1), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        img_dilation = 255-img_dilation
        cv2.imwrite(name, img_dilation)

    def test_img(self):
        # Вирівнювання кольору працює доволі довго :'(
        self.normalize_color_of_img(self.img_name)

        # Стандартизація зображення
        self.transform_img("img/res.jpg")
        self.transform_img("img/res.jpg")

        # висвітлення фону паперу, прибирання малих косяків
        result = self.brightness_n_bw_img()

        # підведення ліній
        result = 255 - result
        self.outline_line_on_img(result, "img/res2.jpg")

        # розбиття на рядки
        size = []
        row_c = self.divide_to(10, 150, "img/res2.jpg", "r", False)
        # розбиття на слова
        for i in range(row_c):
            word_c = self.divide_to(20, 50, "img/r" + (str(i)) + ".jpg", "w", False)
            # розбиття на букви
            size_symbol = []
            for j in range(word_c):
                symbol_c = self.divide_to(5, 1, "img/r" + (str(i)) + "w" + (str(j)) + ".jpg", "s", True)
                size_symbol.append(symbol_c)
            size.append(size_symbol)
        return size

    def is_trash(self, img):
        c = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] < 100:
                    c += 1
        if c > 0.8 * (len(img)*len(img[0])):
            return False
        return True

    def find_near(self, page, box):
        index = []
        for i in range(len(page)):
            lenght = 1000000000
            index.append(0)
            for j in range(len(box)):
                dist = self.distance_to_point(page[i][0], page[i][1], box[j][0], box[j][1])
                if lenght > dist:
                    lenght = dist
                    index[i] = j
        return index

    def distance_to_point(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def aprox_in_array(self, index, page):
        max_i = 100000000
        page_retouch = [[max_i, max_i], [max_i, -max_i], [-max_i, -max_i], [-max_i, max_i]]
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

    def forvard_back(self, arr):
        arr2 = []
        for i in range(len(arr)):
            arr2.append(arr[len(arr) - i - 1])
        return arr2

    def white_black(self, source_name, result_name, brightness):
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
