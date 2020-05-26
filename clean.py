# Cleans rejected offsite images
import os
from module import database

# Paths
offsite_images_main_path = os.getcwd() + '/output/lp_training/offsite_images/'


def app():
    rows = database.get_rejected_offsite_images()
    for row in rows:
        # noinspection PyShadowingBuiltins
        id = row[0]
        file_name_cropped = row[1]
        try:
            # Delete image
            file = str(id) + file_name_cropped
            os.remove(offsite_images_main_path + file)

            # Delete database record
            database.delete_rejected_offsite_image_record(id)

            print('[Info] removed ' + file)
        except AttributeError as e:
            print(e)
