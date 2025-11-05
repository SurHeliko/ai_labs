import os
from PIL import Image
import torch
from torchvision import transforms


def compute_mean_std_recursive_any_size(root_dir, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, f))

    if not image_paths:
        raise ValueError("Не найдено изображений!")

    print(f"Найдено изображений: {len(image_paths)}")

    # Инициализируем накопители
    pixel_sum = torch.zeros(3)
    pixel_sum_sq = torch.zeros(3)
    total_pixels = 0

    to_tensor = transforms.ToTensor()  # [H, W, C] -> [C, H, W], значения в [0,1]

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            tensor = to_tensor(img)  # shape: [3, H, W]
        except Exception as e:
            print(f"⚠️ Пропущено (ошибка): {path} — {e}")
            continue

        # Количество пикселей в этом изображении
        c, h, w = tensor.shape
        num_pixels = h * w
        total_pixels += num_pixels

        # Сумма значений и сумма квадратов по каналам
        pixel_sum += tensor.sum(dim=[1, 2])  # [3]
        pixel_sum_sq += (tensor ** 2).sum(dim=[1, 2])  # [3]

        if (i + 1) % 500 == 0:
            print(f"Обработано: {i + 1} / {len(image_paths)}")

    if total_pixels == 0:
        raise ValueError("Ни одно изображение не было успешно загружено.")

    # Вычисляем mean и std
    mean = pixel_sum / total_pixels
    std = torch.sqrt(pixel_sum_sq / total_pixels - mean ** 2)

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    dataset_path = input("Введите путь к корневой папке датасета: ").strip()
    if not os.path.isdir(dataset_path):
        print("❌ Папка не найдена!")
    else:
        try:
            mean, std = compute_mean_std_recursive_any_size(dataset_path)
            print(f"\n✅ Готово!")
            print(f"Mean: {[round(x, 4) for x in mean]}")
            print(f"Std:  {[round(x, 4) for x in std]}")
        except Exception as e:
            print(f"Ошибка: {e}")