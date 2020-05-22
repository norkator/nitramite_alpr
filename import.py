# This script detects and imports car images
from libraries.nitramite_alpr.module import vehicle_detection
from matplotlib.path import Path
from module import fileutils
from objects import File
import os

# Paths
offsite_images_input = os.getcwd() + '/output/lp_training/offsite_image_input/'  # Dump files to be read in here
offsite_images_output = os.getcwd() + '/output/lp_training/offsite_images/'  # Works like output folder as a whole

# Check path existence
Path(offsite_images_input).mkdir(parents=True, exist_ok=True)
Path(offsite_images_output).mkdir(parents=True, exist_ok=True)


# Return source folder files but from old to last
def get_image_objects():
    image_objects = []
    fileutils.create_directory(offsite_images_input + 'processed/')
    for file_name in fileutils.get_camera_image_names(offsite_images_input):
        if file_name != 'processed' and file_name != 'Thumbs.db' and file_name.find('.lock') is -1:
            image_objects.append(
                File.File(
                    None,
                    offsite_images_input,
                    file_name,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
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
