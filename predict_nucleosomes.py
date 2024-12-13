#!/usr/bin/env python
###created by sjmondo on 2024-11-05 09:29:15.946788###
"""
predict_nucleosomes.py --input in.matrix --outdir path
Predict nucleosomes using method described in Mondo et al., 2025. Input is a data matrix generated using generate_matrix.py, then will write output per scaffold to the specified --outdir
"""
import sys
sys.dont_write_bytecode = True
import os, argparse, pickle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,usage='predict_nucleosomes.py --input in.matrix --outdir path',description='Predict nucleosomes using method described in Mondo et al., 2025. Input is a data matrix generated using generate_matrix.py, then will write output per scaffold to the specified --outdir')
parser.add_argument('-i', '--input', default=None, help='input data matrix, generated using generate_matrix.py.')

if len(sys.argv) < 2:
    parser.print_help()
    exit()

from keras.utils import to_categorical
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import LSTM, GRU
import numpy, math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.layers import Bidirectional
import pandas
import random
from keras.layers import Flatten
import itertools
from keras import optimizers
from keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
import tensorflow as tf
sys.stderr.write('%s\n' % (device_lib.list_local_devices()))
import time
import gc

def load_model(model, weights):
    """loads json formatted neural network pre-computed model"""
    from keras.models import model_from_json
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)
    return loaded_model

def get_data(data_list, x_col_details, y_col):
    x = []
    y = []
    for data in data_list:
        details = []
        for col in x_col_details:
            x_type = col.split('.')[1]
            column = int(col.split('.')[0])
            if x_type == 'categorical':
                vector = [0 for _ in range(5)]
                vector[int(data[column])] = 1
                details.extend(vector)
            else:
                try:
                    details.append(float(data[column]))
                except ValueError:
                    details.append(0.0)
        y.append(float(data[y_col].strip()))
        x.append(details)
    return x, y

def create_dataset(dataset, y, lookback, dtype='nucleosome'):
    dataX = []
    dataY = []
    from numpy import hstack
    input_data = numpy.array(dataset).T
    dataset = input_data[0].reshape(len(input_data[0]),1)
    for feature in input_data[1:]:
        data = feature.reshape(len(feature),1)
        dataset = hstack((dataset, data))
    if dtype == 'nucleosome':
        for i in range(lookback, len(dataset)-lookback):
            dataX.append(dataset[i-lookback:i+lookback])
            dataY.append(y[i])
    else:
        for i in range(lookback, len(dataset)-1):
            dataX.append(dataset[i-lookback:i+1])
            dataY.append(y[i])
    dataX = numpy.array(dataX)
    return numpy.array(dataX), numpy.array(dataY).reshape(len(dataY),1)

def run_mem_efficient(train_dict, train_file, y_type, x_cols, y_col, xscale, look_back, model, weights):
    """This will train your model using your training dictionary and test it on your test dict for plotting"""
    unique = 0
    x_columns = []
    x_column_details = []
    lineages = []
    results = {}
    model = load_model(model, weights)
    if isinstance(train_dict, list):
        train_keys = train_dict
        train_dict = {}
        for key in train_keys:
            train_dict[key] = []
    else:
        train_keys = list(train_dict.keys())
    test_dict_processed = {}
    for key in train_keys:
        lin = key.split('.')[0]
        scf = key.split('.')[1]
    for key in train_keys:
        if key.split('.')[0] not in lineages:
            lineages.append(key.split('.')[0])
    for x_col in x_cols:
        col = int(x_col.split('.')[0])-2
        x_columns.append(col)
        if x_col.split('.')[1] == 'c':
            x_type = 'categorical'
        else:
            x_type = 'regression'
        if x_type == 'categorical':
            x_column_details.append('%s.categorical.%s' % (col, 5))
        else:
            x_column_details.append('%s.%s' % (col, x_type))
    for sample in train_keys:
        sys.stderr.write('processing %s for prediction\n' % (sample))
        try:
            X,y = get_data(train_dict[sample], x_column_details, y_col)
            if xscale != None:
                X, x_scale = scale_data(X, x_column_details, scale=xscale)
            X,y = create_dataset(X, y, look_back)
            sys.stderr.write('%s\n' % (str(X.shape)))
            yhat = model.predict(X)
            results[sample] = [val[0] for val in yhat]
            write_output(results[sample], look_back, sample, 'wig')
            del(yhat)
            del(X)
            del(y)
            gc.collect()
        except ValueError as e:
            sys.stderr.write('ERROR with %s, check contents and ensure everything is correct. Error is: %s\n' % (sample, e))
            continue
    sys.stderr.write('completed prediction on %s\n' % (train_file))
    exit()

def check_lineages_included(data):
    """If multiple organisms are used, it will separate each individually so they can be scaled individually. This will be identified by '.' in dict keys. For example Mereb1.scaffold_1"""
    lineages = []
    final_dataset = {}
    for org in data:
        if '.' in org:
            if org.split('.')[0] not in lineages:
                lineages.append(org.split('.')[0])
                final_dataset[org] = []
    if lineages != []:
        return data, 'single'
    else:
        for scaffold in data:
            organism = scaffold.split('.')[1]
            final_dataset[organism].append(data[scaffold])
        return final_dataset, 'multiple', lineages

def scale_data(data, datacols, scale='StandardScaler', start=0, end=1):
    """Uses StandardScaler, or MinMaxScaler to scale data."""
    series = pandas.DataFrame(data)
    values = series.values
    updated_datacols = []
    loc = 0
    for col in datacols:
        column = int(col.split('.')[0])
        dtype = col.split('.')[1]
        if dtype == 'categorical':
            num_vectors = int(col.split('.')[2])
            for i in range(column, column+num_vectors):
                updated_datacols.append('%s.categorical' % (loc))
                loc += 1
        else:
            updated_datacols.append('%s.regression' % (loc))
            loc += 1
    for col in updated_datacols:
        column = int(col.split('.')[0])
        datatype = col.split('.')[1]
        if datatype == 'regression':
            in_data = values[:,column].reshape(len(values[:,column]),1)
            full_dataset = []
            i = 0
            while i < len(in_data)-10000:
                if scale == 'StandardScaler':
                    scaler = StandardScaler()
                    dataset = scaler.fit_transform(in_data[i:i+10000])
                elif scale == 'MinMaxScaler':
                    scaler = MinMaxScaler(feature_range=(start, end))
                    dataset = scaler.fit_transform(in_data[i:i+10000])
                else:
                    sys.stderr.write('ERROR: incorrect scaler format. Choose from MinMaxScaler or StandardScaler\n')
                    exit()
                full_dataset.extend(list(dataset[:,0]))
                i += 10000
            if scale == 'StandardScaler':
                scaler = StandardScaler()
                dataset = scaler.fit_transform(in_data[i:])
            elif scale == 'MinMaxScaler':
                scaler = MinMaxScaler(feature_range=(start, end))
                dataset = scaler.fit_transform(in_data[i:])
            full_dataset.extend(list(dataset[:,0]))
            values[:,column] = full_dataset
    return values, scaler

def split_matrix_by_first_col(input_file):
    """splits matrix into a dictionary based on names in first column. All input matrices require this col to for sample labeling"""
    data_dict = {}
    data_dict_revised = {}
    orgs = {}
    for row in open(input_file):
        data = row.split()
        if data[0] not in data_dict:
            data_dict[data[0]] = []
            if data[0].split('.')[0] not in orgs:
                orgs[data[0].split('.')[0]] = 0
        data_dict[data[0]].append(data[1:])
    data_dict_revised = data_dict
    sys.stderr.write('organisms detected: %s\n' % (' '.join(orgs)))
    return data_dict_revised

def write_output(modeled_nucleosomes, look_back, sample, outfmt):
    """modeled_nucleosomes is a dictionary structured like so 'scaffold: [value1, value2, ... valueN]'."""
    if isinstance(modeled_nucleosomes, dict):
        for scaffold in modeled_nucleosomes:
            sys.stdout.write('fixedStep chrom=%s start=1 step=1 span=1\n' % (scaffold))
            sys.stdout.write('0.0\n'*(look_back+1))
            for value in modeled_nucleosomes[scaffold]:
                sys.stdout.write('%s\n' % (value))
    elif isinstance(modeled_nucleosomes, list):
        sys.stdout.write('fixedStep chrom=%s start=1 step=1 span=1\n' % (sample))
        sys.stdout.write('0.0\n'*(look_back+1))
        for value in modeled_nucleosomes:
            sys.stdout.write('%s\n' % (value))

if __name__ == "__main__":
    ARGS = parser.parse_args()
    INPUT = ARGS.input
    LOOKBACK = 80
    XSCALER = 'MinMaxScaler'
    MODEL = 'models/nucleosome_prediction_DL.model'
    WEIGHTS = 'models/nucleosome_prediction_DL.weights.h5'
    DATACOLS = ['2.c', '3.r', '4.r', '5.r', '6.r', '7.r', '8.r']
    sys.stderr.write('splitting data by labels in column 1')
    INPUT_DICT = split_matrix_by_first_col(INPUT)
    sys.stderr.write('done splitting data.')
    MAXLEN = 0
    YTYPE = 'regression'
    RESULTSCOL = 1
    for scaff in INPUT_DICT:
        if len(INPUT_DICT[scaff]) > MAXLEN:
            MAXLEN = len(INPUT_DICT[scaff])
    sys.stderr.write('Largest scaffold = %s' % (MAXLEN))
    TRAINKEYS = list(INPUT_DICT.keys())
    run_mem_efficient(INPUT_DICT, INPUT, YTYPE, DATACOLS, RESULTSCOL, XSCALER, LOOKBACK, MODEL, WEIGHTS)
