# -*- coding: utf-8 -*-
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device('cpu')

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4591, 0.4568, 0.4237],
                             std=[0.2793, 0.275, 0.2813])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                                     transform=data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test',
                                                    transform=data_transforms)

    class_names = train_dataset.classes

    batch_size = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    net = torchvision.models.alexnet(pretrained=True)
    # print(net)

    for param in net.parameters():
        param.requires_grad = False

    num_classes = 3

    new_classifier = net.classifier[:-1]  # берем все слой классификатора кроме последнего
    new_classifier.add_module('fc', nn.Linear(4096, num_classes))  # добавляем последним слой с двумя нейронами на выходе
    net.classifier = new_classifier  # меняем классификатор сети

    net = net.to(device)

    # Перейдем к обучению.
    # Зададим количество эпох обучения, функционал потерь и оптимизатор.
    num_epochs = 2
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # создаем цикл обучения и замеряем время его выполнения
    t = time.time()
    save_loss = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # прямой проход
            outputs = net(images)
            # вычисление значения функции потерь
            loss = lossFn(outputs, labels)
            # Обратный проход (вычисляем градиенты)
            optimizer.zero_grad()
            loss.backward()
            # делаем шаг оптимизации весов
            optimizer.step()
            save_loss.append(loss.item())
            # выводим немного диагностической информации
            if i % 100 == 0:
                print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                      str(i) + ' Ошибка: ', loss.item())

    print(time.time() - t)

    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():  # отключим вычисление граиентов, т.к. будем делать только прямой проход
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)  # делаем предсказание по пакету
            _, pred_class = torch.max(pred.data, 1)  # выбираем класс с максимальной оценкой
            correct_predictions += (pred_class == labels).sum().item()

    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            pred = net(inputs)
            _, pred_class = torch.max(pred.data, 1)

            for i in range(inputs.size(0)):
                img = inputs[i].cpu().numpy().transpose((1, 2, 0))
                # Используем те же mean и std, что и при нормализации при обучении
                mean = np.array([0.4591, 0.4568, 0.4237])
                std = np.array([0.2793, 0.275, 0.2813])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                plt.title(f'Predicted: {class_names[pred_class[i]]}')
                plt.axis('off')
                plt.pause(1)
                plt.clf()

if __name__ == '__main__':
    main()