import os

img_folder = "D:/ASDC_internship/Second_task/jpg"

for filename in os.listdir(img_folder):
    if filename.endswith(".jpg"):
        img_number = str(int(filename.split("_")[1].split(".")[0]))

        new_filename = f"image_{img_number}.jpg"

        old_path = os.path.join(img_folder, filename)
        new_path = os.path.join(img_folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} to {new_filename}")
