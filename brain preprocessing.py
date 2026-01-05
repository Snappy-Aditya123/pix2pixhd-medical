import os
from modelfunc import *
#preprocesssing 1


#preprocessing 2
def move_validation_data(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, split_ratio=0.3):
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    exts = ".tif" # as all iamges are .tif
    files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(exts)]
    total = len(files)
    num_to_move = int(total * split_ratio)

    to_move = random.sample(files, num_to_move) #randomly select files to move 
    moved = 0
    for fname in to_move:
        src_img  = os.path.join(train_img_dir, fname)
        src_mask = os.path.join(train_mask_dir, fname)
        dst_img  = os.path.join(val_img_dir, fname)
        dst_mask = os.path.join(val_mask_dir, fname)
        if os.path.exists(src_mask):
            shutil.move(src_img, dst_img)
            shutil.move(src_mask, dst_mask)
            moved += 1
    print(f"\nDone! Moved {moved} pairs.")
    print(f"Train images left: {len(os.listdir(train_img_dir))}")
    print(f"Val images now:    {len(os.listdir(val_img_dir))}")
