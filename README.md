# waste
Скрипт для подбора объёмов отходов с помощью генетического алгоритма и градиентного спуска.
В общем задача такая. На производстве каждый квартал вывозят мусор. Увозя сообщают только общие Массу и объем вывезенного мусора, а в отчётах требуют прописать сколько вывезли каждого вида отхода.
Задача конечно бредовая, но работа есть работа.
Значения приходилось выдумывать из головы. А когда видов отходов много, то подобрать значения вручную так, чтобы сумма объёмов была равна общему объёму, а сумма масс(объёмов умноженых на насыпные плотности) была равна общей массе — сложно.
Этот скрипт писался для хорошего друга. Просто хотелось облегчить его работу.

Скрипт читает виды отходов из файла waste.xlsx. Для примера, файл с 11 видами отходов лежит в репозитории.
В самом скрипте необходимо ввести общие объем и массу. Можете попробовать значения 81 и 30, объем и масса соответственно. Это реальные числа из отчёта.
Далее скрипт попросит ввести вас число итераций для генетического алгоритма. Рекомендую начать с 10 и повышат, если результат вас не устроит.
Скрипт запишет результаты в файл excel. Потому попросит вас ввести имя файла.
На этом всё. Наслаждайтесь =)

Зависимости:

    Python 3.6 и выше,

    Numpy

    Scipy

    Pandas

    xlrd

    Geneticalgorithm2

