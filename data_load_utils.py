import os
import pathlib
from functools import partial

import numpy as np
import tensorflow as tf


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img, img_size):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, img_size)


def process_path(file_path, class_names, img_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, label


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def train_test_val_img_dataset(
    directory="data", seed=1337, test_size=0.2, val_size=0.2, batch_size=32, image_size=(224, 224)
):
    data_dir = pathlib.Path(directory).with_suffix("")

    list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=False)
    image_count = list_ds.cardinality().numpy()
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False, seed=seed)

    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

    train_ds = list_ds.skip(test_size * image_count + val_size * image_count)
    val_ds = list_ds.take(val_size * image_count)
    test_ds = list_ds.skip(val_size * image_count).take(test_size * image_count)

    process_path_p = partial(process_path, class_names=class_names, img_size=image_size)

    train_ds = train_ds.map(process_path_p, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_path_p, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(process_path_p, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, batch_size)
    val_ds = configure_for_performance(val_ds, batch_size)
    test_ds = configure_for_performance(test_ds, batch_size)

    return train_ds, test_ds, val_ds, class_names
