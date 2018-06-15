import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from numpy.random import uniform as rand


def count_of_same_elements(arr):
    counter = collections.Counter(arr)
    return counter


def rotate(arr, n, flag):
    if flag:
        return arr[+n:] + arr[:+n]
    else:
        return arr[-n:] + arr[:-n]


def mean_square_method(num, len_of_vector):
    vector_of_randoms = list()

    str_num = str(num ** 2)
    center = str_num[4:8]
    vector_of_randoms.append(float("0." + center) ** 2)
    while len(vector_of_randoms) != len_of_vector:
        initial_num = vector_of_randoms[len(vector_of_randoms) - 1]
        str_num = str(initial_num ** 2)
        str_num = str_num[2:]
        center_element = int(len(str_num) / 2)
        center = str_num[center_element - 3:center_element + 3]

        vector_of_randoms.append(float("0." + center) ** 2)

    return vector_of_randoms


def bar(arr):
    arr = pd.DataFrame(arr)
    new_arr = list()
    count_of_intervals = int(np.round(1 + math.log(len(arr), 2)))
    count_of_values_in_interval = int(np.round(len(arr) / count_of_intervals))
    for i in range(0, count_of_intervals):
        if i != count_of_intervals:
            new_arr.append(arr[i * count_of_values_in_interval:(i + 1) * count_of_values_in_interval].sum())
        else:
            new_arr.append(arr[i * count_of_values_in_interval:len(arr)].sum)

    return new_arr


def print_bars(not_sorted_vector, name):
    sorted_vector = sorted(not_sorted_vector)

    sns.distplot(sorted_vector)

    plt.title(name)
    plt.xlabel("Интервал разбиения")
    plt.ylabel("Вероятность попадания в интервал")
    plt.show()


def mean_mul(num1, num2, len_of_vector):
    vector_of_randoms = list()

    str_num = str(num1 * num2)
    center = str_num[4:8]

    vector_of_randoms.append(num1)
    vector_of_randoms.append(num2)

    vector_of_randoms.append(float("0." + center))

    while len(vector_of_randoms) != len_of_vector:
        initial_num2 = vector_of_randoms[len(vector_of_randoms) - 2]
        initial_num1 = vector_of_randoms[len(vector_of_randoms) - 1]

        str_num = str(initial_num1 * initial_num2)
        str_num = str_num[2:]

        center_element = int(len(str_num) / 2)
        center = str_num[center_element - 3:center_element + 3]

        vector_of_randoms.append(float("0." + center) ** 2)

    return vector_of_randoms


def mix_method(num, len_of_vector):
    vector_of_randoms = list()

    while len(vector_of_randoms) != len_of_vector:
        inithial_str = str(bin(num))[2:]

        a = int(rotate(inithial_str, 2, True), 2)
        b = int(rotate(inithial_str, 2, False), 2)

        num = a + b

        vector_of_randoms.append(float("0." + str(num)))

    return vector_of_randoms


def linear_method(k, r0, b, M, len_of_vector):
    vector_of_randoms = list()

    while len(vector_of_randoms) != len_of_vector:
        r = (k * r0 + b) % M

        vector_of_randoms.append(float("0." + str(r)))

        r0 = r

    return vector_of_randoms


def clt(len_of_vector):
    l = len_of_vector
    return (rand(0, 1, l) + rand(0, 1, l) + rand(0, 1, l) + rand(0, 1, l) + rand(0, 1, l)) - 1.5


def gauss(a, b, len_of_vector):
    vector_of_randoms = list()

    for i in range(0, len_of_vector):
        vector_of_randoms.append(random.gauss(a, b))
    return vector_of_randoms


def muler(len_of_vector):
    vector_of_randoms = list()

    while len(vector_of_randoms) != len_of_vector:
        vector_of_randoms.append(math.sqrt(-2 * math.log(rand(0, 1, 1))) * math.cos(2 * math.pi * rand(0, 1, 1)))
    return vector_of_randoms


def norm(x, phi, sigma):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * (math.e ** ((-((x - phi) ** 2)) / (2 * (sigma ** 2))))


def monte_carlo(phi, sigma, len_of_vector):
    vector_of_randoms = list()
    # max = 1 / math.sqrt(2 * math.pi * sigma)
    i = 0
    while len(vector_of_randoms) != len_of_vector:
        x = rand(0, 6, 1) * sigma - 3 * sigma + phi
        y = rand(0, 1 / math.sqrt(2 * math.pi), 1)
        if y <= norm(x, phi, sigma):
            vector_of_randoms.append(x)
            i += 1
    return vector_of_randoms


def bar_disp(current, normal):
    arr_p = list()
    for i in range(0, len(normal)):
        arr_p.append(((current[i] / normal[i]) - 1) / 100)
    return np.mean(arr_p)


if __name__ == "__main__":
    vec_mean_square = mean_square_method(0.2152, 1000)
    vec_mean_mul = mean_mul(0.2152, 0.8245, 1000)
    vec_mix_method = mix_method(2152, 1000)
    vec_linear = linear_method(1220703125, 7, 7, 2147483647, 1000)

    vec_standart_numpy_rnd = np.random.uniform(0, 1, 1000)

    print(np.round(np.mean(vec_standart_numpy_rnd), 3))
    print(np.round(np.var(vec_standart_numpy_rnd), 3))
    print(bar_disp(vec_mix_method, vec_standart_numpy_rnd))

    print_bars(vec_mean_square, "Метод Серединных квадратов")
    print_bars(vec_mean_mul, "Метод Серединных произведений")
    print_bars(vec_mix_method, "Метод Перемешивания")
    print_bars(vec_linear, "Линейный конгруэнтный метод")
    print_bars(vec_standart_numpy_rnd, "Нормальное Библиотечное распределние")

    print_bars(clt(1000), "Центральная предельная теорема")
    print_bars(muler(1000), "Метод Мюлера")
    print_bars(monte_carlo(0, 1, 1000), "Метод Монте-Карло")
    print_bars(gauss(0, 1, 1000), "Гауссовское распределение")
