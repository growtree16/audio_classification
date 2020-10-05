#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:26:04 2020

@author: Hannah Xu
"""
import tensorflow as tf
import pathlib
import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
audio = os.path.join(dir_path, 'animal_sounds')
audio_ds = pathlib.Path(audio)
list_ds = tf.data.Dataset.list_files(str(audio_ds/'*/*'))

unique_labels = ['dog', 'cat', 'bird']

index = 0
record = {}
for label in unique_labels:
    record[label] = index
    index += 1
    
    
df = pd.DataFrame(columns = ['file_path', 'label'])
for f in list_ds:
    temp = str(f.numpy())
    label = record[temp.split('/')[-2]]
    file_name = temp[1:]
    row = pd.Series([file_name, label], index = ['file_path', 'label'])
    df = df.append(row, ignore_index=True)  
print (df.shape)
    
    
df.to_csv('audio_list.csv')