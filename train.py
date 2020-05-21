from libraries.nitramite_alpr.module import detect_model, data_generator
from pathlib import Path
import tensorflow as tf
import numpy as np
import os

# Paths
base_path = os.getcwd() + '/output/lp_training/'
model_output_path = base_path + 'model/'

# Check path existence
Path(model_output_path).mkdir(parents=True, exist_ok=True)

# Get train images count
train_dir = base_path + 'train/'
NUM_TRAIN_SAMPLES = len([name for name in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, name))])

# Training variables
BATCH_SIZE = 28  # original 32, 32 crash/oom with RTX, 28 'magic number' works well
BATCH_SIZE_VAL = 10  # original 10
NUM_EPOCHS = 100
INITIAL_EPOCH = 0
NUM_STEPS = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
NUM_VAL_SAMPLES = 150
NUM_STEPS_VAL = int(NUM_VAL_SAMPLES / BATCH_SIZE_VAL)
LR_INIT = 0.001
lr_decay = 0.5
decay_every = int(NUM_EPOCHS / 5)


def train():
    model = detect_model.detect_model()
    lr_v = tf.Variable(LR_INIT)
    adam_p = tf.optimizers.Adam(lr_v, beta_1=0.9)
    val_loss = 100

    for epoch in range(NUM_EPOCHS):
        train_generator = data_generator.data_generator(base_path + 'train/', BATCH_SIZE)
        val_generator = data_generator.data_generator(base_path + 'test/', BATCH_SIZE_VAL)

        loss = []

        for step in range(NUM_STEPS):
            batch_images, batch_label = train_generator.__next__()
            with tf.GradientTape(persistent=True) as tape:
                predict_label = model(batch_images)
                loss_mse = tf.reduce_mean(tf.losses.mean_squared_error(predict_label, batch_label))
                loss.append(loss_mse)
            grad = tape.gradient(loss_mse, model.trainable_weights)
            adam_p.apply_gradients(zip(grad, model.trainable_weights))
        loss_val = []

        for step in range(NUM_STEPS_VAL):
            batch_images, batch_label = val_generator.__next__()
            predict_label = model(batch_images)
            loss_mse = tf.reduce_mean(tf.losses.mean_squared_error(predict_label, batch_label))
            loss_val.append(loss_mse)
        val_loss_cur = np.mean(loss_val)

        # Lower test validation loss is what matters
        print('[Info] Epoch: %d, training loss: %f, test validation loss: %f' % (epoch, np.mean(loss), val_loss_cur))

        if val_loss_cur <= val_loss:
            val_loss = val_loss_cur
            model.save_weights(model_output_path + 'lp_model.h5')
            print('[info] model savepoint...')
        if (epoch != 0) and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            lr_v.assign(LR_INIT * new_lr_decay)
