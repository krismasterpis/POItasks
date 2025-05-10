import os
from PIL import Image

def extract_texture_patches(input_folder_name: str, output_folder_name: str, patch_size: int = 128):

    if input_folder_name is None:
        input_path = os.curdir
    else:
        input_path = input_folder_name
    output_path = output_folder_name
    os.makedirs(output_path, exist_ok = True)
    print(f"Katalog wyjściowy: '{output_path}'")

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for image_file in os.listdir(input_path):
        image_file_name, image_file_extension = os.path.splitext(image_file)
        if image_file_extension in allowed_extensions:
            category_name, image_file_suffix = os.path.splitext(image_file)
            category_output_path = output_path+"/"+category_name
            os.mkdir(category_output_path)

            print(f"\nPrzetwarzanie obrazu: '{image_file}'")

            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
                    print(f"  Rozmiar oryginału: {img_width} x {img_height}")

                    if img_width < patch_size or img_height < patch_size:
                        print(f"  Ostrzeżenie: Obraz '{image_file}' jest mniejszy niż rozmiar próbki!")
                        continue

                    patch_count = 0
                    # Pętla y (wiersze)
                    for y in range(0, img_height - patch_size + 1, patch_size):
                        # Pętla x (kolumny)
                        for x in range(0, img_width - patch_size + 1, patch_size):
                            box = (x, y, x + patch_size, y + patch_size)

                            patch = img.crop(box)

                            patch_filename = f"{category_name}_y{y}_x{x}{image_file_suffix}"
                            patch_save_path = category_output_path+"/"+patch_filename

                            patch.save(patch_save_path)
                            patch_count += 1

                    if patch_count > 0:
                         print(f"  Wycięto i zapisano {patch_count} próbek.")
                    else:
                         print(f"  Nie wycięto żadnych próbek z obrazu '{image_file}'.")

            except Exception as e:
                print(f"  Błąd podczas przetwarzania pliku '{image_file}': {e}")
        # elif os.path.isfile(image_file):
        #     print(f"\nBład rozszerzenia pliku!")

if __name__ == "__main__":
    extract_texture_patches(None,"output")