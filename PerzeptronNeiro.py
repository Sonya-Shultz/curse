import cv2
import numpy as np


class PerzeptronNeiro:
    size = 28

    def __init__(self, symbol_numb):
        self.mul = [[0 for i in range(PerzeptronNeiro.size)] for j in range(PerzeptronNeiro.size)]
        self.weight = []
        self.limit = 28
        self.sum = 0
        self.symbol_numb = symbol_numb

    def mul_sum_calc(self, img):
        self.sum = 0
        for x in range(self.size):
            for y in range(self.size):
                self.mul[x][y] = img[x][y] * self.weight[x][y]
                self.sum = self.sum + self.mul[x][y]

    def rez(self):
        if self.sum >= self.limit:
            return True
        else:
            return False

    def read_neiro(self):
        if 1039 < self.symbol_numb < 1066 or 1069 < self.symbol_numb < 1072 or (chr(self.symbol_numb) in "ЬҐЄЇІ"):
            self.weight = np.load(chr(self.symbol_numb) + ".npy")
        if 1071 < self.symbol_numb < 1098 or 1111 < self.symbol_numb < 1134 or chr(self.symbol_numb) in "ґьєії":
            self.weight = np.load(chr(self.symbol_numb).upper() + "s.npy")

    def save_neiro(self):
        if 1039 < self.symbol_numb < 1066 or 1069 < self.symbol_numb < 1072 or (chr(self.symbol_numb) in "ЬҐЄЇІ"):
            np.save(chr(self.symbol_numb), self.weight)
        if 1071 < self.symbol_numb < 1098 or 1111 < self.symbol_numb < 1134 or chr(self.symbol_numb) in "ґьєії":
            np.save(chr(self.symbol_numb).upper() + "s", self.weight)

    def incW(self, img):
        for x in range(self.size):
            for y in range(self.size):
                self.weight[x][y] += img[x][y]

    def decW(self, img):
        for x in range(self.size):
            for y in range(self.size):
                self.weight[x][y] -= img[x][y]

    def is_right(self, ex, answer, img): # self - ім'я нейрона, ex - реальна буква, answer - правда, якщо АІ вважає, що це однакові букви
        # if chr(self.symbol_numb) == ex and answer:
        # if chr(self.symbol_numb) != ex and !answer:
        if chr(self.symbol_numb) != ex and answer:
            self.decW(img)
        if chr(self.symbol_numb) == ex and not answer:
            self.incW(img)


def create_pez_neiro():
    # 178-255
    for a in range(1040, 1066):
        new_neir(chr(a), "")
        new_neir(chr(a), "s")
    for a in range(1070, 1072):
        new_neir(chr(a), "")
        new_neir(chr(a), "s")
    ukr_let = "ЬҐЄЇІ"
    for let in ukr_let:
        new_neir(let, "")
        new_neir(let, "s")


def new_neir(ch, end_ch):
    help_arr = []
    for i in range(PerzeptronNeiro.size):
        help_1d = []
        for j in range(PerzeptronNeiro.size):
            help_1d.append(0)
        help_arr.append(help_1d)
    np.save(ch + end_ch, help_arr)


def normalize_input(img):
    help_arr = []
    for i in range(PerzeptronNeiro.size):
        help_1d = []
        for j in range(PerzeptronNeiro.size):
            help_1d.append(1) if img[i][j] > 220 else help_1d.append(0)
        help_arr.append(help_1d)
    return help_arr


def recognize(neiro, n_img):
    neiro.mul_sum_calc(n_img)
    print("it is  " + chr(neiro.symbol_numb)) if neiro.rez() else print("It is NOT " + chr(neiro.symbol_numb))
    return neiro.sum, neiro.rez()
