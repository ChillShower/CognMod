import os
import shutil

def filter_and_copy_images(
    source_folder,
    dest_folder,
    age_range=(0, 116),
    genders=[0, 1],
    races=[0, 1, 2, 3, 4]
):
    """
    Filters images by age, gender, race and copies them to a destination folder.

    Parameters:
        source_folder (str): Path to the source image folder
        dest_folder (str): Path to destination folder
        age_range (tuple): (min_age, max_age) inclusive
        genders (list): Allowed genders (0=male, 1=female)
        races (list): Allowed races (0=White,1=Black,2=Asian,3=Indian,4=Others)
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    count = 0
    for file in os.listdir(source_folder):
        if not file.lower().endswith(".jpg"):
            continue

        try:
            # parse filename: [age]_[gender]_[race]_[datetime].jpg
            parts = file.split("_")
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
        except Exception:
            # skip badly formatted files
            continue

        if age_range[0] <= age <= age_range[1] and gender in genders and race in races:
            src_path = os.path.join(source_folder, file)
            dst_path = os.path.join(dest_folder, file)
            shutil.copy2(src_path, dst_path)  # copy with metadata
            count += 1

    print(f"✅ {count} images copied to {dest_folder}")


# Example usage
if __name__ == "__main__":
    source = "../archive/UTKFace"  # folder with images
    dest = "Filtered_Faces"  # folder to copy into
    filter_and_copy_images(
        source, dest,
        age_range=(75, 80),   # only 20–40 years old
        genders=[0,1],          # only male
        races=[0]          # White & Asian
    )
