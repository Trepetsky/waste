total_volume = float(input("Общий объём: "))
total_mass = float(input("Общая масса: "))
name_file_excel = input("Название excel файла: ")
import numpy as np
from scipy.optimize import minimize
import xlrd
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga
from time import perf_counter

def f(X):
    fine_mass = np.abs(total_mass - np.sum(X*density))*10
    fine_volume = np.abs(total_volume - np.sum(X))*5
    fine_50 = []
    for i in range(number):
        if density[i]<1:
            fine_50.append(X[i])
# = np.sum(np.abs(density_100_50[i] + X[i] * density_total_volume_100[i]))
    fine_50 = (max(fine_50) - min(fine_50))*5
    if fine_volume < 1 and fine_mass < 1 and fine_volume == 0:
        return (fine_volume + fine_mass + fine_50) * fine_mass
    elif fine_volume < 1 and fine_mass < 1 and fine_mass == 0:
        return (fine_volume + fine_mass + fine_50) * fine_volume
    elif fine_volume < 1 and fine_mass < 1:
        return (fine_volume + fine_mass + fine_50) * fine_volume * fine_mass
    else:
        return fine_volume + fine_mass + fine_50





start = perf_counter()###################

waste = pd.read_excel("Waste.xlsx")
waste = np.array(waste)
name = []
density = []
volume_approximate = []
for i in np.arange(np.shape(waste)[0]):
    name.append(waste[i][0])
    density.append(waste[i][1])
    volume_approximate.append(waste[i][2])

number = np.shape(waste)[0]
density=np.array(density)
density_100 = density*100
density_total_volume_100 = density*100 / total_volume
density_100_50 = 50 - density_100

varbound = np.array([0] * 2 * number).reshape(number, 2)
for i in np.arange(number):
    varbound[i][1] = volume_approximate[i]

algorithm_param = {'max_num_iteration': 1000, \
                   'population_size': 5000, \
                   'mutation_probability': 0.1, \
                   'elit_ratio': 0.01, \
                   'crossover_probability': 0.5, \
                   'parents_portion': 0.3, \
                   'crossover_type': 'uniform', \
                   'max_iteration_without_improv': 20000}

X = np.array([0.3, 9, 38, 1.90, 0.8, 0.5, 18.87, 10.75])

model = ga(function=f, dimension=number, variable_type='real', variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

model.run(no_plot=True)

def Optimale(X):
    fine_mass = np.abs(total_mass - np.sum(np.abs(X)*density))*10
    fine_volume = np.abs(total_volume - np.sum(np.abs(X)))*5
    fine_50 = np.array([0]*number)
    for i in range(number):
        if density[i]<1:
            fine_50[i] = np.sum(np.abs(density_100_50[i] + X[i] * density_total_volume_100[i]))
        else:
            fine_50[i]=0
    fine_50 = np.sum(fine_50)*5
    for i in X:
        if i > total_volume/2:
            fine_50 += i
    if fine_volume < 1 and fine_mass < 1 and fine_volume == 0:
        return (fine_volume + fine_mass + fine_50) * fine_mass
    elif fine_volume < 1 and fine_mass < 1 and fine_mass == 0:
        return (fine_volume + fine_mass + fine_50) * fine_volume
    elif fine_volume < 1 and fine_mass < 1:
        return (fine_volume + fine_mass + fine_50) * fine_volume * fine_mass
    else:
        return fine_volume + fine_mass + fine_50

X = np.array(model.output_dict["variable"])
res = minimize(Optimale, X, method='nelder-mead', options={'xtol': 1e-8, "maxfev": 20, 'disp': True})
for i in range(100):
    X = np.array(res.x)
    res = minimize(Optimale, X, method='nelder-mead', options={'xtol': 1e-8, "maxfev": 20, 'disp': True})
excel = [0] * 3 * (number + 3)
temp = list(range(0, (number + 3) * 3, 3))
for i in range(0, number):
    excel[temp[i]] = name[i]
excel[(number + 1) * 3] = "Общий объём"
excel[(number + 2) * 3] = "Общая масса"
print(number + 1)
excel = np.array(excel).reshape(number + 3, 3)
for i in range(number):
    excel[i][1] = res.x[i]
    excel[i][2] = res.x[i] * density[i]
excel[number + 1][1] = np.sum(res.x)
excel[number + 2][1] = np.sum(res.x * density)
excel = pd.DataFrame(excel)
excel = excel.replace("0", np.nan)
for i in range(number + 3):
    excel[1][i] = float(excel[1][i])
    excel[2][i] = float(excel[2][i])
excel.columns = ["Название", "Объём", "Масса"]
excel.to_excel(f'{name_file_excel}.xlsx')

end = perf_counter()
for i in range(number):
    print(name[i], model.output_dict["variable"][i], model.output_dict["variable"][i] * density[i])
finish_total_mass = [0] * number
for i in range(number):
    finish_total_mass[i] = model.output_dict["variable"][i] * density[i]
print("Общий объём", np.sum(model.output_dict["variable"]))
print("Общая масса", np.sum(finish_total_mass))
print((end-start)/60, "мин.")