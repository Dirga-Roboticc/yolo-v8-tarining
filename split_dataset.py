import os
import random
import shutil
import argparse
from tqdm import tqdm

def split_dataset(root_dir, split_ratio):
    # Tentukan direktori untuk gambar dan label
    source_images_dir = os.path.join(root_dir, 'images')
    source_labels_dir = os.path.join(root_dir, 'labels')

    # Periksa apakah direktori sumber ada
    if not os.path.exists(source_images_dir) or not os.path.exists(source_labels_dir):
        raise FileNotFoundError("Direktori sumber gambar atau label tidak ditemukan.")

    # Buat direktori train dan val di dalam root_dir
    train_images_dir = os.path.join(root_dir, 'train', 'images')
    train_labels_dir = os.path.join(root_dir, 'train', 'labels')
    val_images_dir = os.path.join(root_dir, 'val', 'images')
    val_labels_dir = os.path.join(root_dir, 'val', 'labels')

    # Buat folder train dan val jika belum ada
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Ambil semua file gambar dari direktori sumber
    images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Acak urutan file gambar
    random.shuffle(images)

    # Tentukan jumlah data untuk train dan val
    train_size = int(len(images) * split_ratio)

    # Bagi dataset menjadi train dan val
    train_images = images[:train_size]
    val_images = images[train_size:]

    def copy_files(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
        for image in tqdm(image_list, desc="Copying files"):
            image_name, image_ext = os.path.splitext(image)
            label = image_name + '.txt'
            
            try:
                shutil.copy(os.path.join(src_img_dir, image), dst_img_dir)
                shutil.copy(os.path.join(src_lbl_dir, label), dst_lbl_dir)
            except FileNotFoundError as e:
                print(f"Warning: File not found - {e}")
            except shutil.SameFileError:
                print(f"Warning: Source and destination are the same file - {image}")
            except PermissionError:
                print(f"Warning: Permission denied when copying - {image}")

    # Pindahkan file ke folder train dan val
    copy_files(train_images, source_images_dir, source_labels_dir, train_images_dir, train_labels_dir)
    copy_files(val_images, source_images_dir, source_labels_dir, val_images_dir, val_labels_dir)

    print(f"Dataset successfully split: {len(train_images)} train, {len(val_images)} val.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train and val sets.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing images and labels folders')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of train set (default: 0.8)')

    args = parser.parse_args()

    split_dataset(args.root_dir, args.split_ratio)
