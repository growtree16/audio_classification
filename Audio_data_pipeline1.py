#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:54:54 2020

@author: local
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import io_ops
import pandas as pd
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
    data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename}).audio.flatten()
    #print (len(data))
    #print (data.dtype)
    return data

def get_dataset(filepath):
    df = pd.read_csv(filepath)
    df['feature_ds'] = df['file_path'].apply(lambda x: x[1:-1]).apply(load_wav_file)
    num_of_features = 10000#len(df.head(1).feature_ds.values[0])
    columns = []
    for i in range(num_of_features):
        column = 'feature_ds_' + str(i)
        columns.append(column)
    df_feature = pd.DataFrame(df['feature_ds'].to_list()).loc[:,:9999]
    df_feature.columns=columns[:10000]
    df_feature['label'] = df['label']
    
    return df_feature

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)])
    optimizer=keras.optimizers.Adam(learning_rate=100)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        
    return model


def get_compiled_CNN_model():
    model = models.Sequential()
    model.add(tf.keras.layers.Reshape((100, 100, 1), input_shape=(10000,)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.summary()
    optimizer=keras.optimizers.Adam(learning_rate=10000)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        
    return model


def main():
    

    ds = get_dataset('audio_list500.csv')

    target = ds.pop('label')
    dataset = tf.data.Dataset.from_tensor_slices((ds.values, target.values))
    train_dataset = dataset.shuffle(len(ds)).batch(1)
    print (dataset)
    print (train_dataset)
    
    model = get_compiled_CNN_model()
    history = model.fit(train_dataset, epochs=15)
    # plot loss during training
    plt.figure(figsize = (16, 6))
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()