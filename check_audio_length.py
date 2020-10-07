#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:54:16 2020

@author: Hannah Xu
"""

import audio_data_pipeline1 as audio_data_pipeline1
import tensorflow as tf
import pathlib
import os
import pandas as pd


def get_dataset_length(df):
    df['audio_length'] = df['file_path'].apply(lambda x: x[1:-1]).apply(audio_data_pipeline1.load_wav_file).apply(lambda x: len(x))
    
    return df


dir_path = os.path.dirname(os.path.realpath(__file__))
audio = os.path.join(dir_path, 'animal_sounds')
audio_ds = pathlib.Path(audio)
list_ds = tf.data.Dataset.list_files(str(audio_ds/'*/*'))
print (list_ds)
unique_labels = ['dog', 'cat', 'bird', '_background_noise_']

index = 0
record = {}
for label in unique_labels:
    record[label] = index
    index += 1
    
    
df = pd.DataFrame(columns = ['file_path', 'label'])
for f in list_ds.take(500):
    temp = str(f.numpy())
    label = record[temp.split('/')[-2]]
    file_name = temp[1:]
    row = pd.Series([file_name, label], index = ['file_path', 'label'])
    df = df.append(row, ignore_index=True)  

df = get_dataset_length(df)  
    
df.to_csv('audio_length_list500.csv', index = False)

