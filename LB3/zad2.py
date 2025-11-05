import math
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

def main():
    df = pd.read_csv('data.csv')
    y = df.iloc[:, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    X = df.iloc[:, [0, 1, 2, 3]].values

    features = torch.tensor(X, dtype=torch.float)
    answers = torch.tensor(y, dtype=torch.float)
    answers = answers.reshape(-1, 1)

    print(f"Признаки:\n{features}")
    print(f"Отыеты:\n{answers}")

    linear = nn.Linear(4, 1)
    lossFn = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    for i in range(0,10):
        pred = linear(features)
        loss = lossFn(pred, answers)
        print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
        loss.backward()
        optimizer.step()

    # посчитаем ошибки
    sum_err = 0
    for feature, answer in zip(features, answers):
        predict = linear(feature)
        if(predict > 0):
            pred = 1
        else:
            pred = -1
        sum_err += abs((math.ceil(answer) - pred) / 2)

    print("Всего ошибок: ", sum_err)

if __name__ == '__main__':
    main()