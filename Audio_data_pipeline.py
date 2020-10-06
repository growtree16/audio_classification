#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:34:46 2020

@author: Hannah Xu
@Credit to: https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1
"""

import pandas as pd
import tensorflow as tf



AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(df):
    file_path_ds = tf.data.Dataset.from_tensor_slices(df.file_path)
    label_ds = tf.data.Dataset.from_tensor_slices(df.label)
    return tf.data.Dataset.zip((file_path_ds, label_ds))


def load_audio(file_path, label):
    # Load one second of audio at 44.1kHz sample-rate
    audio = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=160)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    # Repeat dataset forever
    ds = ds.repeat()
    # Prepare batches
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def get_compiled_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),tf.keras.layers.Dense(10, activation='relu'),\
                                         tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        
    return model


def main():
    # Load meta.csv containing file-paths and labels as pd.DataFrame
    df = pd.read_csv('audio_list100.csv')
    
    ds = get_dataset(df)
    train_ds = prepare_for_training(ds)
    
    print (train_ds)

    batch_size = 64
    train_steps = len(df) / batch_size
    
    #Define Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(160, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = get_compiled_model()
    model.fit(train_ds, epochs=100, steps_per_epoch=train_steps)


if __name__ == '__main__':
    main()
