import os
import random
import numpy as np
from scipy.optimize import minimize
import xlrd
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga

# вводим общее количество объёма и массы
total_volume = float(input("Общий объём: "))
total_mass = float(input("Общая масса: "))
# устанавливаем число итераций для генетического алгоритма
num_iteration = int(input("Число попыток: "))
# вводим название эксель файла для pandas
name_file_excel = input("Название excel файла: ")

# проверяем существует файл с таким названием или нет
answer = "42"
while True:
    if name_file_excel + ".xlsx" in os.listdir(path="."):
        if answer == "42":
            answer = input("Файл с таким именем уже существует, хотите перезаписать его? Введите \"да\" или \"нет\": ")
        if answer.lower() in ["нет", "н", "no", "n"]:
            name_file_excel = input("Название excel файла: ")
            if name_file_excel + ".xlsx" in os.listdir(path="."):
                answer = input(
                    "Файл с таким названием уже существует, хотите перезаписать его? Введите \"да\" или \"нет\": ")
            else:
                break
        elif answer.lower() in ["да", "д", "yes", "y"]:
            break
        else:
            answer = input("Неверный ввод. Введите \"да\" или \"нет\": ")
    else:
        break


# вычитается в fine_light, чтобы объёмы лёгких отходов не были равны друг другу
# random, как гарант того, что при каждым запуске скрипта получатся разные разности объёмов
random_in_fine_light = random.uniform(3, 8)


# функция для начисления штрафа GA, она возвращает сумму штрафов, а GA пытается эту сумму минимизировать
# X - объёмы отходов в массиве numpy
def fine_GA(volume):
    # штраф для объёма, округление нужно потому что конечный ответ
    # должен быть с тремя знаками после запятой (в geneticalgorithm2 реализованы только float64)
    # хоть GA и генерирует объёмы в float64 и округление банковское,
    # но это не важно, GA всё-равно находит правильные значея
    fine_volume = abs(total_volume - np.sum(np.around(volume, 3))) * 100000
    # аналогично только для массы
    fine_mass = abs(total_mass - np.sum(np.around(volume * density, 3))) * 200000
    # штрафы для лёгких и тяжёлых отходов, чтобы объёмы лёгких отхоходов распределялись равномернее,
    # а объёмы тяжёлых не стремились к нулю
    fine_light_list = []
    fine_heavy_list = []
    for i in range(number):
        if density[i] < 0.8:
            fine_light_list.append(volume[i])
        else:
            fine_heavy_list.append(volume[i])
    fine_light = abs(max(fine_light_list) - random_in_fine_light - min(fine_light_list)) * 20000
    # здесь округление используется, чтобы быстрее максимизировать объёмы тяжёлых отходов
    fine_heavy = (5 - round(min(fine_heavy_list), 1)) * 500000
    return fine_volume + fine_mass + fine_light + fine_heavy


# чтение из эксель файла наименований, плотностей и примерных объёмов отходов
waste = pd.read_excel("Waste.xlsx")
waste = np.array(waste)
name = []
density = []
volume_approximate = []
for i in np.arange(np.shape(waste)[0]):
    name.append(waste[i][0])
    density.append(waste[i][1])
    volume_approximate.append(waste[i][2])

# преобразование списка в массив вне функции для ускорения расчётов
density = np.array(density)

# ограничение знычений объёмов для GA,
# чтобы он не генерировал значения больше, чем примерный объём отхода до вывоза
number = np.shape(waste)[0]
varbound = np.array([0] * 2 * number).reshape(number, 2)
for i in range(number):
    varbound[i][1] = volume_approximate[i]

# параметры GA и запуск, см. документацию https://github.com/PasaOpasen/geneticalgorithm2
algorithm_param = {'max_num_iteration': num_iteration,
                   'population_size': 20000,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.05,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': 500}

model = ga(function=fine_GA, dimension=number, variable_type='real', variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

model.run(no_plot=True)


# функция для градиентного спуска
def fine_minimize(volume):
    fine_mass = np.abs(total_mass - np.sum(np.around(volume * density, 3))) * 10
    fine_volume = np.abs(total_volume - np.sum(np.around(volume, 3))) * 5
    # жесткое ограничение, чтобы не получались значения < 0
    if min(volume) < 0:
        fine_volume *= 10000000
    return fine_volume + fine_mass


# запуск градиентного спуска
volume_GA = np.array(model.output_dict["variable"])
res = minimize(fine_minimize, volume_GA, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

# запись в эксель файл
excel = [0] * 3 * (number + 3)
temp = list(range(0, (number + 3) * 3, 3))
for i in range(0, number):
    excel[temp[i]] = name[i]
excel[(number + 1) * 3] = "Общий объём"
excel[(number + 2) * 3] = "Общая масса"
excel = np.array(excel).reshape(number + 3, 3)
for i in range(number):
    excel[i][1] = round(res.x[i], 3)
    excel[i][2] = round(res.x[i] * density[i], 3)
excel[number + 1][1] = np.sum(np.around(res.x, 3))
excel[number + 2][1] = np.sum(np.around(res.x * density, 3))
excel = pd.DataFrame(excel)
excel = excel.replace("0", np.nan)
for i in range(number + 3):
    excel[1][i] = float(excel[1][i])
    excel[2][i] = float(excel[2][i])
excel.columns = ["Название", "Объём", "Масса"]
excel.to_excel(f'{name_file_excel}.xlsx')
ccc = pd.read_excel(f'{name_file_excel}.xlsx')