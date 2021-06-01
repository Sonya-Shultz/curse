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
            right_chr = '_'
            max_sum = -50000
            max_el = all_letters[0]
            for el in all_letters:
                el_sum, right = pn.recognize(el, img2)
                if max_sum < el_sum:
                    right_chr = chr(el.symbol_numb)
                    max_sum = el_sum
                    max_el = el
                el.is_right(let, right, img2)
            max_el.is_right(let, True, img2)
            print(right_chr, end=" ")
            counter += 1
    print("")


def out_line_all(path_big, path_small):
    s_let = pn.PerzeptronNeiro.c_small
    s_big = pn.PerzeptronNeiro.c_big
    img_h = rt.RetouchNSplitImg("img/test11.jpg")
    for let in s_big:
        if let not in "": #"ЯЬЩЮ":
            path = path_big + "1\\" + let + "\\"
            images = glob.glob(path + "*.jpg")
            for q in range(len(images)):
                img2 = Image.open(images[q])
                img2 = np.array(img2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img_h.outline_line_on_img(img2, images[q])

    for let in s_let.upper():
        if let not in "": #"ЯЬЩЮ":
            if let in s_big:
                let2 = let.lower()
                path = path_small + "1\\" + let2 + "\\"
                images = glob.glob(path + "*.jpg")
                for q in range(len(images)):
                    img2 = Image.open(images[q])
                    img2 = np.array(img2)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img_h.outline_line_on_img(img2, images[q])
                    print(images[q])
            else:
                let2 = let.lower()
                path = path_big + "1\\" + let + "\\"
                images = glob.glob(path + "*.jpg")
                for q in range(len(images)):
                    img2 = Image.open(images[q])
                    img2 = np.array(img2)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img_h.outline_line_on_img(img2, images[q])


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
    all_letters = whole_brain()
    w = 0
    for i in range(len(size)):
        for j in range(len(size[i])):
            for a in range(size[i][j]):
                img2 = Image.open("img/r" + str(i) + "w" + str(j) + "s" + str(a) + ".jpg")
                img2 = np.array(img2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = pn.normalize_input(img2)
                max_sum = -50000
                for el in all_letters:
                    el_sum, right = pn.recognize(el, img2)
                    if el_sum > max_sum and right:
                        max_sum = el_sum
                    el.is_right(ex[w], right, img2)
                w += 1
    for neiro in all_letters:
        neiro.save_neiro()


def recognize_prez_neiro(size):
    all_letters = []
    for q in range(1000, 1500):
        if chr(q) in pn.PerzeptronNeiro.c_big:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
            check(one_letter.weight, chr(q))
            print(chr(q))
        if chr(q) in pn.PerzeptronNeiro.c_small:
            one_letter = pn.PerzeptronNeiro(q)
            one_letter.read_neiro()
            all_letters.append(one_letter)
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
                max_sum = -50000
                char = ' '
                right = False
                num = 0
                while num < len(all_letters):
                    el_sum, right = pn.recognize(all_letters[num], img2)
                    if max_sum < el_sum: #and right:
                        max_sum = el_sum
                        char = chr(all_letters[num].symbol_numb)
                        print(max_sum, char, end=" | ")
                    num += 1
                print(" ")
                all_text += char
            all_text += " "
        all_text += "\n"
    print(all_text)


def check(arr, name):
    help_2d = []
    if name == name.lower():
        name = name + "s"
    for a in arr:
        help_1d = []
        for r in a:
            #print(r, end=" ")
            if r < -125:
                help_1d.append((0, 0, 0))
            else:
                if r > 130:
                    help_1d.append((255, 255, 255))
                else:
                    help_1d.append((125 + r, 125 + r, 125 + r))
        help_2d.append(help_1d)
        #print("")
    help_2d = np.array(help_2d, dtype=np.uint8)
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
    all_letters = "ґ"
    filtr = rt.RetouchNSplitImg("img/test1.jpg")
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
            name = new_path + (image.rpartition('.')[0])[5:len(image.rpartition('.')[0])] + ".jpg"
            bg.save(name)
            name = name.replace("\\", '/')
            filtr.white_black(name, name, 0.95)
            filtr.divide_to(10, 10, name, "t", True)
        print(el)


def create_folders():
    for el in pn.PerzeptronNeiro.c_small: #.upper():
        os.mkdir("c_newsmall/"+el)


def hellp_acc(path, all_letters, time):
    images = glob.glob(path + "*.jpg")
    time = time % len(images)
    img2 = Image.open(images[time])
    img2 = np.array(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = pn.normalize_input(img2)
    right_chr = '_'
    max_sum = 0
    max_el = all_letters[0]
    for el in all_letters:
        el_sum, right = pn.recognize(el, img2)
        if max_sum < el_sum:
            right_chr = chr(el.symbol_numb)
            max_sum = el_sum
            max_el = el
    return right_chr


def accuracy_of_execution(path_small, path_big, time):
    s_let = pn.PerzeptronNeiro.c_small
    s_big = pn.PerzeptronNeiro.c_big
    all_letters = whole_brain()
    w = 0
    for let in s_big:
        path = path_big + "\\" + let + "\\"
        r_ch = hellp_acc(path, all_letters, time)
        #print(r_ch, let)
        if r_ch in let:
            w += 1
    for let in s_let.upper():
        if let in s_big:
            let2 = let.lower()
            path = path_small + "\\" + let2 + "\\"
            r_ch = hellp_acc(path, all_letters, time)
            #print(r_ch, let2)
            if r_ch in let2:
                w += 1
        else:
            let2 = let.lower()
            path = path_big + "\\" + let + "\\"
            r_ch = hellp_acc(path, all_letters, time)
            #print(r_ch, let2)
            if r_ch in let2:
                w += 1
    return w/(len(s_big)+len(s_let))


def normal_w():
    all_letters = whole_brain()
    for i in all_letters:
        i.normalise_weight()
        i.save_neiro()


def whole_brain():
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
    return all_letters


def teach_more(path):
    images = glob.glob(path + "/*.jpg")
    ex = ["Вбити", "розкромсати", "розчленити", "Але", "поті м", "я", "згадую", "що", "я", "ді вчинка", "пі втора",
          "мет ри", "зростом", "ледве", "ношУ", "паке и         ",  "з", "магаазинн у", "і", "сплю", "з", "рукою",
          "пді ", "п ошкою"]
    for iw in range(len(images)):
        try:
            img_t = rt.RetouchNSplitImg(path+"/"+str(iw+1)+".jpg")
            size_t = img_t.test_img()
            #recognize_prez_neiro(size_t)
            teach_prez_neiro(size_t, ex[iw])
            print(iw)
        except:
            print("An exception occurred")




if __name__ == '__main__':
    # create_folders()
    #clear_pictures("*.png",  "c_big\\", "c_newbig")
    #clear_pictures("*.jpg", "c_big\\", "c_newbig")
    # clear_pictures("*.jpg", "c_sml\\", "c_newsmall")
    #out_line_all("c_newbig", "c_newsmall")
    #create_prez_neiro()
    #normal_w()
    #img = rt.RetouchNSplitImg("img/test9.jpg")
    #size = img.test_img()
    #for i in range(20):
        #teach_more("img_for_teach")
    img = rt.RetouchNSplitImg("img/test16.jpg")
    size = img.test_img()
    #for i in range(7):
        #teach_prez_neiro(size, "Автострада")
        #teach_prez_neiro(size, "неходитудитамтбечекютьнеприє мності нуякжетудинехо тивонижчеаютькошенянаімя Га ав                                            ")
        #teach_prez_neiro(size, "коли на бекр тьзсамогоранкуати  щенав непрокинувсятисея лю")
        #teach_prez_neiro(size, "колинатебекричатьзсамогоранкУатищенавітьнеп кинувсятисеясплю")
    recognize_prez_neiro(size)
    ser = []
    for i in range(73, 78):
        ser.append(accuracy_of_execution("c_newsmall", "c_newbig", i))
       # print(ser[len(ser)-1])
    print("середнє: " + str(np.mean(ser)))
