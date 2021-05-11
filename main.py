import RetouchNSplitImg as rt
import PerzeptronNeiro as pn
import cv2


# ex - рядок з віповдями без пробілів типу: "БабаВаляшизануласьВонакупила3кгоселедцю"
def teach_prez_neiro(size, ex):
    all_letters = []
    for q in range(1000, 1500):
        if 1039 < q < 1066 or 1069 < q < 1072 or (chr(q) in "ЬҐЄЇІ"):
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
        if 1071 < q < 1098 or 1111 < q < 1134 or (chr(q) in "ґьєії"):
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
    w = 0
    for i in range(len(size)):
        for j in range(len(size[i])):
            for a in range(size[i][j]):
                img2 = pn.normalize_input(cv2.imread("r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg", cv2.IMREAD_GRAYSCALE))
                for el in all_letters:
                    el_sum, right = pn.recognize(el, img2)
                    if el_sum > max_sum and right:
                        max_sum = el_sum
                    el.is_right(ex[w], right, img2)
                w += 1


def recognize_prez_neiro(size):
    all_letters = []
    for q in range(1000, 1500):
        if 1039 < q < 1066 or 1069 < q < 1072 or (chr(q) in "ЬҐЄЇІ"):
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
        if 1071 < q < 1098 or 1111 < q < 1134 or (chr(q) in "ґьєії"):
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
    for i in range(len(size)):
        for j in range(len(size[i])):
            for a in range(size[i][j]):
                img2 = pn.normalize_input(cv2.imread("r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg", cv2.IMREAD_GRAYSCALE))
                max_sum = 0
                char = ''
                for el in all_letters:
                    el_sum, right = pn.recognize(el, img2)
                    if el_sum > max_sum and right:
                        max_sum = el_sum
                        char = chr(el.symbol_numb)
                print(char, end="")


if __name__ == '__main__':
    img = rt.RetouchNSplitImg("test3.jpg")
    size = img.test_img()
    # ex = зчитуєм з файлу чи просто прописуємо в ручну букви, але дані мають бути ідеальними для навчання
    ex = ""
    teach_prez_neiro(size, ex)
