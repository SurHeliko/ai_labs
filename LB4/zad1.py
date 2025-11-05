import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),  # слой линейных сумматоров
                                    nn.Tanh(),  # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )

    # прямой проход
    def forward(self, X):
        pred = self.layers(X)
        return pred

def main():
    df = pd.read_csv('dataset_simple.csv')
    y = df.iloc[:, 2].values
    y = np.where(y == 1, 1, -1)
    X = df.iloc[:, [0, 1]].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    features = torch.tensor(X, dtype=torch.float)
    answers = torch.tensor(y, dtype=torch.float)
    answers = answers.reshape(-1, 1)

    inputSize = features.shape[1]  # количество признаков задачи
    hiddenSizes = 3  # число нейронов скрытого слоя
    outputSize = 1

    # Создаем экземпляр нашей сети
    net = NNet(inputSize, hiddenSizes, outputSize)

    # Для обучения нам понадобится выбрать функцию вычисления ошибки
    lossFn = nn.MSELoss()

    # и алгоритм оптимизации весов
    # при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
    # и скорость обучения
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    epohs = 100
    for i in range(0, epohs):
        pred = net.forward(features)  # прямой проход - делаем предсказания
        loss = lossFn(pred, answers)  # считаем ошибу
        optimizer.zero_grad()  # обнуляем градиенты
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())

    # Посчитаем ошибку после обучения
    with torch.no_grad():
        pred = net.forward(features)

    pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
    err = torch.sum(torch.abs(answers - pred) / 2).item()
    print('\nОшибка (количество несовпавших ответов): ')
    print(err)  # обучение работает, не делает ошибок или делает их достаточно мало

if __name__ == '__main__':
    main()

# Пасхалка, кто найдет и сможет объяснить, тому +
# Тут написано аналитическое решение задачи линейной регрессии методом наименьших квадратов (МНК)

# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])
#
# y = df.iloc[:, -1].values
#
# w = np.linalg.inv(X.T @ X) @ X.T @ y
#
# predicted = X @ w
#
# print(predicted)