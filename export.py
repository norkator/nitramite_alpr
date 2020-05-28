from objects import TrainingFile
from module import database
from pathlib import Path
import glob
import cv2
import os

# Paths
output_root_folder_path = os.getcwd() + '/output/'
output_lp_training_main_path = os.getcwd() + '/output/lp_training/'
output_lp_training_lp_path = output_lp_training_main_path + 'lp/'

output_lp_training_train_path = output_lp_training_main_path + 'train/'
output_lp_training_train_labels_path = output_lp_training_train_path + 'labels/'

output_lp_training_test_path = output_lp_training_main_path + 'test/'
output_lp_training_test_labels_path = output_lp_training_test_path + 'labels/'

# Check path existence
Path(output_lp_training_main_path).mkdir(parents=True, exist_ok=True)
Path(output_lp_training_lp_path).mkdir(parents=True, exist_ok=True)
Path(output_lp_training_train_path).mkdir(parents=True, exist_ok=True)
Path(output_lp_training_train_labels_path).mkdir(parents=True, exist_ok=True)
Path(output_lp_training_test_path).mkdir(parents=True, exist_ok=True)
Path(output_lp_training_test_labels_path).mkdir(parents=True, exist_ok=True)


def delete_old_files(path):
    files = glob.glob(path + '*')
    print('[Info] found ' + str(len(files)) + ' files. Deleting...')
    for f in files:
        try:
            os.remove(f)
        except PermissionError as e:
            pass


# Original images are resize'd,
# bounding box must be adjusted
def calculate_resize_bbox(orig_w, orig_h, new_w, new_h, orig_x, orig_y, orig_x2, orig_y2):
    new_x = int((new_w / orig_w) * orig_x)
    new_y = int((new_h / orig_h) * orig_y)
    new_x2 = int((new_w / orig_w) * orig_x2)
    new_y2 = int((new_h / orig_h) * orig_y2)

    r = [new_x, new_y, new_x2, new_y2]
    print(r)
    return r


# Write training or testing images with labels
def write_image_and_label_files(path, image_name, image_data, x, y, x2, y2):
    try:
        cv2.imwrite(path + image_name, image_data)
        with open((path + 'labels/' + image_name + '.txt'), 'w') as text_file:
            text_file.write(str(x) + ',' + str(y) + ',' + str(x2) + ',' + str(y2))
    except Exception as e:
        print(e)


def app():
    print('[Info] deleting old files')
    delete_old_files(output_lp_training_lp_path)
    delete_old_files(output_lp_training_train_path)
    delete_old_files(output_lp_training_train_labels_path)
    delete_old_files(output_lp_training_test_path)
    delete_old_files(output_lp_training_test_labels_path)

    rows = database.get_labeled_for_training_lp_images()
    lp_training_image_objects = []
    for row in rows:
        # Get db row fields
        # noinspection PyShadowingBuiltins
        id = row[0]
        label = row[1]
        file_name_cropped = row[2]
        labeling_image_x = row[3]
        labeling_image_y = row[4]
        labeling_image_x2 = row[5]
        labeling_image_y2 = row[6]

        # There's small differences in naming between OI and off site images
        input_image = None
        cropped_name = None
        if label is not None:
            cropped_name = file_name_cropped
            input_image = output_root_folder_path + label + '/' + file_name_cropped
        else:
            cropped_name = str(id) + file_name_cropped
            input_image = output_root_folder_path + 'lp_training/offsite_images/' + str(id) + file_name_cropped

        # Make objects
        lp_training_image_object = TrainingFile.TrainingFile(
            id, cropped_name, label,
            labeling_image_x, labeling_image_y, labeling_image_x2, labeling_image_y2,
            input_image
        )
        lp_training_image_objects.append(lp_training_image_object)

    # Export process
    if len(lp_training_image_objects) > 0:
        modulo = (len(lp_training_image_objects) / 150)
        index = 0
        for tio in lp_training_image_objects:
            try:
                img = cv2.imread(tio.file_full_path)

                original_h, original_w, original_c = img.shape
                new_size = 256  # will be square, same width, same height
                resize = cv2.resize(img, (new_size, new_size))
                new_bbox = calculate_resize_bbox(
                    original_w, original_h, new_size, new_size,
                    tio.labeling_image_x, tio.labeling_image_y, tio.labeling_image_x2, tio.labeling_image_y2
                )

                if index % modulo:
                    # Training
                    write_image_and_label_files(
                        output_lp_training_train_path, tio.file_name_cropped, resize,
                        new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]
                    )
                else:
                    # Testing
                    write_image_and_label_files(
                        output_lp_training_test_path, tio.file_name_cropped, resize,
                        new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]
                    )

                # Write cropped lp's just for possible inspection
                lp_crop = img[tio.labeling_image_y:tio.labeling_image_y2, tio.labeling_image_x:tio.labeling_image_x2]
                try:
                    cv2.imwrite(output_lp_training_lp_path + '/' + tio.file_name_cropped, lp_crop)
                except Exception as e:
                    print(e)

                '''
                cv2.imshow("OrigImg", img)
                cv2.imshow("256Img", resize)
                cv2.imshow("LpCrop", lp_crop)
                cv2.waitKey(0)
                '''
            except AttributeError as e:
                print(e)
            index = index + 1

        print('[info] export completed')
    else:
        print('No export actions to process')
