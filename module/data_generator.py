import threading
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import random


class ThreadSafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def thread_safe_generator(f):
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


@thread_safe_generator
def data_generator(base_path, batch_size):
    # Provide the same seed
    input_files = [f for f in listdir(base_path) if isfile(join(base_path, f))]
    input_files = sorted(input_files)
    gt_path = base_path + 'labels/'

    length = len(input_files)
    file_count = 0
    index = [i for i in range(length)]
    random.shuffle(index)
    while True:
        if file_count + batch_size > length:
            break
        ind = index[file_count:file_count + batch_size]
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            n = ind[i]
            img_name = input_files[n]
            inputs = Image.open(join(base_path, img_name))
            inputs = np.array(inputs, dtype=np.float32)
            inputs = inputs / 255.0

            batch_x += [inputs]
            txt_name = img_name + '.txt'
            f = open(join(gt_path, txt_name))
            gt = f.readline()
            f.close()
            gt = gt.split(',')
            gt_labels = []
            for n in gt:
                gt_labels.append(float(n) / 256.0)
            gt_labels = np.array(gt_labels)

            batch_y += [gt_labels]
            file_count += 1
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        yield (batch_x, batch_y)
