import os
from PIL import Image

def convert_webp_to_jpg_and_delete(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Папка '{folder_path}' не существует.")
        return

    converted_count = 0
    error_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.webp'):
            webp_path = os.path.join(folder_path, filename)
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(folder_path, jpg_filename)

            try:
                with Image.open(webp_path) as img:
                    # Обработка прозрачности
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Сохранение в JPG
                    img.save(jpg_path, 'JPEG', quality=95)

                # Удаляем исходный .webp файл только после успешного сохранения JPG
                os.remove(webp_path)
                print(f"✅ {filename} → {jpg_filename} (исходный файл удалён)")
                converted_count += 1

            except Exception as e:
                print(f"❌ Ошибка при конвертации {filename}: {e}")
                error_count += 1

    print(f"\nГотово! Успешно конвертировано: {converted_count}, ошибок: {error_count}")

if __name__ == "__main__":
    folder = input("Введите путь к папке с .webp изображениями: ").strip()
    if folder:
        convert_webp_to_jpg_and_delete(folder)
    else:
        print("Путь не указан.")