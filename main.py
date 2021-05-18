import RetouchNSplitImg as rt
import PerzeptronNeiro as pn
import cv2
import glob
from PIL import Image
import os
import numpy as np
import random


def find_all_for_teach(path, all_letters, let, one_count, attempt):
    images = glob.glob(path + "*.jpg")
    counter = 0
    for q in range(len(images)):
        if counter < one_count:
            help_q = (one_count*attempt + q) % len(images)
            img2 = Image.open(images[help_q])
            img2 = np.array(img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = pn.normalize_input(img2)
            right_chr = ' '
            for el in all_letters:
                el_sum, right = pn.recognize(el, img2)
                if right:
                    right_chr = chr(el.symbol_numb)
                el.is_right(let, right, img2)
            print(right_chr, end=" ")
            counter += 1
    print("")


def teach_prez_neiro_new(one_count, attempt):
    all_letters = []
    s_let = list(pn.PerzeptronNeiro.c_small)
    s_big = list(pn.PerzeptronNeiro.c_big)
    random.shuffle(s_let)
    random.shuffle(s_big)
    s_let = ''.join(s_let)
    s_big = ''.join(s_big)
    print(s_big, s_let)
    path_big = "c_newbig"
    path_small = "c_newsmall"
    for let in s_big:
        q = ord(let)
        one_letter = pn.PerzeptronNeiro(q)
        one_letter = one_letter.read_neiro()
        all_letters.append(one_letter)
    for let in s_let.upper():
        '''if let not in s_big:
            q = ord(let.lower())
            one_letter = pn.PerzeptronNeiro(q)
            one_letter = one_letter.read_neiro()
            all_letters.append(one_letter)
        else:'''
        q = ord(let.lower())
        one_letter = pn.PerzeptronNeiro(q)
        one_letter = one_letter.read_neiro()
        all_letters.append(one_letter)
    for let in s_big:
        if let not in "": #"ЯЬЩЮ":
            path = path_big + "\\" + let + "\\"
            find_all_for_teach(path, all_letters, let, one_count, attempt)
    for let in s_let.upper():
        if let not in "": #"ЯЬЩЮ":
            if let in s_big:
                let2 = let.lower()
                path = path_small + "\\" + let2 + "\\"
                find_all_for_teach(path, all_letters, let2, one_count, attempt)
            else:
                let2 = let.lower()
                path = path_big + "\\" + let + "\\"
                find_all_for_teach(path, all_letters, let2, one_count, attempt)
    for neiro in all_letters:
        neiro.save_neiro()


# ex - рядок з віповдями без пробілів типу: "БабаВаляшизануласьВонакупила3кгоселедцю"
def teach_prez_neiro(size, ex):
    all_letters = []
    for q in range(1000, 1500):
        if chr(q) in pn.PerzeptronNeiro.c_big:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
        if chr(q) in pn.PerzeptronNeiro.c_small:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
    w = 0
    for i in range(len(size)):
        for j in range(len(size[i])):
            for a in range(size[i][j]):
                img2 = pn.normalize_input(cv2.imread("r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg", cv2.IMREAD_GRAYSCALE))
                max_sum = 0
                for el in all_letters:
                    el_sum, right = pn.recognize(el, img2)
                    if el_sum > max_sum and right:
                        max_sum = el_sum
                    el.is_right(ex[w], right, img2)
                w += 1


def recognize_prez_neiro(size):
    all_letters = []
    for q in range(1000, 1500):
        if chr(q) in pn.PerzeptronNeiro.c_big:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
            #print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                             #for row in one_letter.weight]))
            check(one_letter.weight, chr(q))
            print(chr(q))
        if chr(q) in pn.PerzeptronNeiro.c_small:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
            #print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                             #for row in one_letter.weight]))
            check(one_letter.weight, chr(q))
            print(chr(q))
    all_text = ""
    for i in range(len(size)):
        for j in range(len(size[i])):
            for a in range(size[i][j]):
                img2 = Image.open("img/r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg")
                img2 = np.array(img2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = pn.normalize_input(img2)
                # img2 = pn.normalize_input(cv2.imread("r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg",
                #   cv2.IMREAD_GRAYSCALE))
                max_sum = 0
                char = ' '
                right = False
                num = 0
                while num < len(all_letters):
                    el_sum, right = pn.recognize(all_letters[num], img2)
                    if max_sum < el_sum and chr(all_letters[num].symbol_numb) not in "":
                        max_sum = el_sum
                        char = chr(all_letters[num].symbol_numb)
                        print(max_sum, char, end=" | ")
                    num += 1
                print(" ")
                all_text += char
                #print(char, end="")
            all_text += " "
            #print(" ", end="")
        all_text += "\n"
        #print("")
    print(all_text)


def check(arr, name):
    help_2d = []
    if name == name.lower():
        name = name + "s"
    for a in arr:
        help_1d = []
        for r in a:
            if r < 0:
                help_1d.append((0, 0, 0))
            if r == 0:
                help_1d.append((125, 125, 125))
            if r > 0:
                help_1d.append((255, 255, 255))
            #help_1d.append((0, 0, 0)) if r < 0 else help_1d.append((255,255,255))
        help_2d.append(help_1d)
    help_2d = np.array(help_2d, dtype=np.uint8)
    #help_2d = np.array(help_2d)
    #print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in help_2d]))
    new_image = Image.fromarray(help_2d)
    new_image.save(name + '.jpg')



def create_prez_neiro():
    for q in range(1000, 1500):
        if chr(q) in pn.PerzeptronNeiro.c_big:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.new_neir()
            one_letter.save_neiro()
        if chr(q) in pn.PerzeptronNeiro.c_small:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.new_neir()
            one_letter.save_neiro()


def clear_pictures(s_format, old, new):
    # all_letters = "абвгґдеєжзиіїйклмнпростуфхцчшщьюя".upper()
    all_letters = "йії"
    filtr = rt.RetouchNSplitImg("img/test1.jpg")
    # all_letters = "А"
    for el in all_letters:
        path = old + el + "\\"
        new_path = new
        images = glob.glob(path + s_format)
        for image in images:
            img = Image.open(image)
            bg = img
            if s_format == "*.jpg":
                bg = img
            else:
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, img)
            # bg.resize((pn.PerzeptronNeiro.size, pn.PerzeptronNeiro.size))
            name = new_path + (image.rpartition('.')[0])[5:len(image.rpartition('.')[0])] + ".jpg"
            bg.save(name)
            name = name.replace("\\", '/')
            filtr.white_black(name, name, 0.95)
            filtr.divide_to(10, 10, name, "t", True)
        print(el)


def create_folders():
    for el in pn.PerzeptronNeiro.c_small: #.upper():
        os.mkdir("c_newsmall/"+el)


if __name__ == '__main__':

    # ex = зчитуєм з файлу чи просто прописуємо в ручну букви, але дані мають бути ідеальними для навчання
    # ex = ""
    #create_prez_neiro()
    # create_folders()
    # clear_pictures("*.png",  "c_big\\", "c_newbig")
    # clear_pictures("*.jpg", "c_big\\", "c_newbig")
    # clear_pictures("*.jpg", "c_sml\\", "c_newsmall")
    # teach_prez_neiro(size, ex)
    #for i in range(0, 200):
        #teach_prez_neiro_new(1, i)
    img = rt.RetouchNSplitImg("img/test10.jpg")
    size = img.test_img()
    print(size)
    recognize_prez_neiro(size)
