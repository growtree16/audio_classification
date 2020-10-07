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
from sklearn.model_selection import train_test_split
#import random
#random.seed(1)


def get_dataset_filepath_and_label(dir_path):
    audio = os.path.join(dir_path, 'speech_commands_v0.01')
    unique_labels = [item for item in os.listdir(audio) if os.path.isdir(os.path.join(audio, item))]
    audio_ds = pathlib.Path(audio)
    list_ds = tf.data.Dataset.list_files(str(audio_ds/'*/*'))
    
    index = 0
    record = {}
    for label in unique_labels:
        record[label] = index
        index += 1        
        
    df = pd.DataFrame(columns = ['file_path', 'label'])
    for f in list_ds.take(500):
        filepath = str(f.numpy())
        label = record[filepath.split('/')[-2]]
        file_name = filepath[1:]
        row = pd.Series([file_name, label], index = ['file_path', 'label'])
        df = df.append(row, ignore_index=True) 
    
    
    return df

def get_dataset_filepath_and_label_train_test_split(dir_path, test_size = 0.3):
    audio = os.path.join(dir_path, 'speech_commands_v0.01')
    unique_labels = [item for item in os.listdir(audio) if os.path.isdir(os.path.join(audio, item))]
    audio_ds = pathlib.Path(audio)
    list_ds = tf.data.Dataset.list_files(str(audio_ds/'*/*'))
    
    index = 0
    record = {}
    for label in unique_labels:
        record[label] = index
        index += 1        
        
    df = pd.DataFrame(columns = ['file_path', 'label'])
    for f in list_ds.take(50):
        filepath = str(f.numpy())
        label = record[filepath.split('/')[-2]]
        file_name = filepath[1:]
        row = pd.Series([file_name, label], index = ['file_path', 'label'])
        df = df.append(row, ignore_index=True) 
        
    df_train, df_test = train_test_split(df, test_size=test_size)
    
    return df_train, df_test


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = get_dataset_filepath_and_label(dir_path)    
    df.to_csv('audio_list_500.csv', index = False)
    df_train, df_test = get_dataset_filepath_and_label_train_test_split(dir_path)
    df_train.to_csv('audio_list_train.csv')
    df_test.to_csv('audio_list_test.csv')

if __name__ == '__main__':
    main()