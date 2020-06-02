from shutil import copyfile
import os

NUM_IMGS_PER_CLASS = 10

INPUT_DIR = "imgs_out"

OUTPUT_DIR = 'imgs_out_sub'

os.mkdir(OUTPUT_DIR)

for fol in ['train', 'test']:
    os.mkdir(OUTPUT_DIR + "/" + fol)

    for i in range(10):
        classname = "c" + str(i)
        os.mkdir(OUTPUT_DIR + "/" + fol + "/" + classname)
        imgs_out_dir = INPUT_DIR + "/" + fol + "/" + classname

        count = 0
        for img_name in os.listdir(imgs_out_dir):
            src = INPUT_DIR + "/" + fol + "/" + classname + "/" + img_name
            dst = OUTPUT_DIR + "/" + fol + "/" + classname + "/" + img_name
            copyfile(src, dst)

            count += 1
            if count == NUM_IMGS_PER_CLASS:
                break
