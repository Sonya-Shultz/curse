import numpy as np


class PerzeptronNeiro:
    size = 28
    c_big = "АБВГҐДЕЗІЇПРТУФЧ"
    c_small = "абвгдеєжзиіїйклмнпростуфхцчшщьюя"

    def __init__(self, symbol_numb):
        self.mul = [[0 for i in range(PerzeptronNeiro.size)] for j in range(PerzeptronNeiro.size)]
        self.weight = []
        self.limit = 500
        self.sum = 0
        self.symbol_numb = symbol_numb

    def mul_sum_calc(self, img):
        self.sum = 0
        for x in range(self.size):
            for y in range(self.size):
                self.mul[x][y] = img[x][y] * self.weight[x][y]
                self.sum = self.sum + self.mul[x][y]
                if img[x][y] == 0 and self.weight[x][y] > 0.02 * self.limit: #0:
                    self.mul[x][y] -= int(0.3 * self.weight[x][y])
                    self.sum = self.sum + self.mul[x][y]

    def rez(self):
        if self.sum >= self.limit:
            return True
        else:
            return False

    def read_neiro(self):
        if chr(self.symbol_numb) in PerzeptronNeiro.c_big:
            self.weight = np.load("neiro/" + chr(self.symbol_numb).upper() + ".npy")
        if chr(self.symbol_numb) in PerzeptronNeiro.c_small:
            self.weight = np.load("neiro/" + chr(self.symbol_numb).upper() + "s.npy")
        return self

    def save_neiro(self):
        if chr(self.symbol_numb) in PerzeptronNeiro.c_big:
            np.save("neiro/" + chr(self.symbol_numb).upper(), self.weight)
        if chr(self.symbol_numb) in PerzeptronNeiro.c_small:
            np.save("neiro/" + chr(self.symbol_numb).upper() + "s", self.weight)

    def incW(self, img):
        for x in range(self.size):
            for y in range(self.size):
                self.weight[x][y] += img[x][y]

    def decW(self, img):
        for x in range(self.size):
            for y in range(self.size):
                self.weight[x][y] -= img[x][y]

    def is_right(self, ex, answer, img): # self - ім'я нейрона, ex - реальна буква, answer - правда, якщо АІ вважає, що це однакові букви
        if chr(self.symbol_numb) != ex and answer:
            self.decW(img)
        if chr(self.symbol_numb) == ex and not answer:
            self.incW(img)

    def new_neir(self):
        self.weight = [[0 for i in range(PerzeptronNeiro.size)] for j in range(PerzeptronNeiro.size)]


def normalize_input(img):
    help_arr = []
    for i in range(PerzeptronNeiro.size):
        help_1d = []
        for j in range(PerzeptronNeiro.size):
            help_1d.append(0) if img[i][j] > 120 else help_1d.append(1)
        help_arr.append(help_1d)
    return help_arr


def recognize(neiro, n_img):
    neiro.mul_sum_calc(n_img)
    # print(neiro.sum)
    # print("it is  " + chr(neiro.symbol_numb)) if neiro.rez() else print("It is NOT " + chr(neiro.symbol_numb))
    return neiro.sum, neiro.rez()
