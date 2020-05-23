# This script detects and imports car images
from libraries.nitramite_alpr.module import vehicle_detection
from module import fileutils
from pathlib import Path
from objects import File
import os

# Paths
offsite_images_input = os.getcwd() + '/output/lp_training/offsite_image_input/'  # Dump files to be read in here
offsite_images_output = os.getcwd() + '/output/lp_training/offsite_images/'  # Works like output folder as a whole

# Check path existence
Path(offsite_images_input).mkdir(parents=True, exist_ok=True)
Path(offsite_images_output).mkdir(parents=True, exist_ok=True)


# Return off site training images
def get_image_objects():
    image_objects = []
    fileutils.create_directory(offsite_images_input + 'processed/')
    for file_name in fileutils.get_camera_image_names(offsite_images_input):
        if file_name != 'processed' and file_name != 'Thumbs.db' and file_name.find('.lock') is -1:
            gm_time = fileutils.get_file_create_time(offsite_images_input, file_name)
            image_objects.append(
                File.File(
                    None,
                    offsite_images_input,
                    file_name,
                    fileutils.get_file_extension(offsite_images_input, file_name),
                    fileutils.get_file_create_year(gm_time),
                    fileutils.get_file_create_month(gm_time),
                    fileutils.get_file_create_day(gm_time),
                    fileutils.get_file_create_hour(gm_time, 0),
                    fileutils.get_file_create_minute(gm_time),
                    fileutils.get_file_create_second(gm_time),
                    fileutils.get_file_mtime(offsite_images_input, file_name)
                )
            )
    return image_objects


def app():
    for image_object in get_image_objects():
        try:
            # Also runs all sub processes
            vehicle_detection.analyze_image(
                offsite_images_output,
                image_object
            )
        except Exception as e:
            print(e)
