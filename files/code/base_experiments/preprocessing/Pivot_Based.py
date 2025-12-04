import csv
import os
import re
import json
import math
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path

class Instance:
    def __init__(self, index, values, label):
        self.index = index
        self.values = values
        self.values_old = values.copy()
        self.label = label

    def get_at(self, index):
        return self.values[index]

    def set_at(self, index, value):
        self.values[index] = value

    def dist_cat(self, other, is_categorical, num_distinct_values):
        value = 0
        for j in range(len(self.values)):
            if is_categorical[j] and self.values[j] != other.values[j]:
                value += 1 - 1 / math.pow(num_distinct_values[j], 2)  # contribuição baseada em probabilidade
        return math.sqrt(value)

class Relation:
    def __init__(self, header, num_rows, num_cols, instances, name, has_label, is_col_categorical, maps_cat_cols):
        self.name = name
        self.header = header
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.instances = instances
        self.has_label = has_label
        self.is_normalized = False
        self.is_col_categorical = is_col_categorical
        self.maps_cat_cols = maps_cat_cols

    @staticmethod
    def read_csv(filename, label_col, has_header):
        df = pd.read_csv(filename, low_memory=False)
        sep = ','
        if len(df.columns) < 2:
            sep = ';'
            df = pd.read_csv(filename, sep=None)
        if label_col is None:
            label_col = len(df.columns)-1
        n_rows = len(df)#Relation.getNumRows(filename)
        n_data_cols = Relation.getNumCols(filename, sep)
        has_label = False

        #if has_header:
        #    n_rows -= 2

        if label_col >= 0:
            has_label = True
            n_data_cols -= 1

        is_col_categorical = [False] * n_data_cols
        instances = [None] * n_rows
        header = None

        with open('config_dataset.json', 'r') as config:
            dtypes = config.read()
        dataset_dtypes = json.loads(dtypes)

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=sep)
            if has_header:
                header = next(reader)  # skip column names
                #v = next(reader)  # read column types
                v = Relation.get_type(filename, dataset_dtypes, n_data_cols)
                for j in range(n_data_cols):
                    if v[j] == "cat" or v[j] == "object":
                        is_col_categorical[j] = True

            df = Relation.convert_string_to_number(df, v)
            #for i, row in enumerate(reader):
            for i, row in df.iterrows():
                values = [0.0] * n_data_cols
                label = -1
                for j, val in enumerate(row):
                    if j == label_col:
                        label = int(val) if isinstance(val, int) or (isinstance(val, str) and val.isnumeric()) else (0 if val == 'no' else 1)
                    else:
                        values[j] = float(val)

                instances[i] = Instance(i, values, label)

        maps_cat_cols = [defaultdict(int) if is_col_categorical[j] else None for j in range(n_data_cols)]

        return Relation(header, n_rows, n_data_cols, instances, Relation.getName(filename), has_label, is_col_categorical, maps_cat_cols)

    @staticmethod
    def convert_string_to_number(df, types):
        for i, t in enumerate(types):
            if t == 'object':
                values = {k:j for j, k in enumerate(df.iloc[:,i].unique())}
                df.iloc[:,i].replace(values, inplace=True)
        return df



    @staticmethod
    def get_type(dataset, config, num_col):
        file = re.sub(r'_v[0-9]{1,2}', '', dataset.split(os.sep)[-1].split('.')[0])
        if file in config:
            dtypes = config[file]
            if '...' in dtypes:
                tipo = dtypes[0]
                return [tipo] * num_col
            else:
                return dtypes
        # print(f'Not set type {dataset}')
        return []

    @staticmethod
    def getName(filename):
        return Path(filename).stem

    @staticmethod
    def getNumCols(filename, sep):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=sep)
            header = next(reader)
            return len(header)

    @staticmethod
    def getNumRows(filename):
        with open(filename, newline='') as csvfile:
            return sum(1 for _ in csvfile)

    def normalize(self):
        assert not self.is_normalized
        assert self.num_rows > 0
        assert self.num_cols > 0
        assert self.instances is not None

        for j in range(self.num_cols):
            if not self.is_col_categorical[j]:
                min_val = float('inf')
                max_val = float('-inf')
                for i in range(self.num_rows):
                    min_val = min(self.instances[i].get_at(j), min_val)
                    max_val = max(self.instances[i].get_at(j), max_val)
                for i in range(self.num_rows):
                    value = (self.instances[i].get_at(j) - min_val) / (max_val - min_val)
                    self.instances[i].set_at(j, value)

        self.process_cat_cols()  # convert categorical columns into continuous ones, and normalize them

        self.is_normalized = True

    def count_frequencies(self):
        for j in range(self.num_cols):
            if self.is_col_categorical[j]:
                for i in range(self.num_rows):
                    self.maps_cat_cols[j][self.instances[i].get_at(j)] += 1

    def minimum_description_length(self, sorted_frequencies):
        cut_point = -1
        minimum_description_length = float('inf')
        for i in range(len(sorted_frequencies)):
            description_length = 0

            pre_average = sum(sorted_frequencies[:i]) / i if i > 0 else 0
            post_average = sum(sorted_frequencies[i:]) / (len(sorted_frequencies) - i)

            if i > 0:
                description_length += math.log2(1 + pre_average)
            description_length += math.log2(1 + post_average)

            for j in range(i):
                description_length += math.log2(1 + abs(pre_average - sorted_frequencies[j]))
            for j in range(i, len(sorted_frequencies)):
                description_length += math.log2(1 + abs(post_average - sorted_frequencies[j]))

            if description_length < minimum_description_length:
                cut_point = i
                minimum_description_length = description_length

        return cut_point

    def process_cat_cols(self):
        # get metadata about the categorical columns
        self.count_frequencies()
        num_cat_cols = 0
        num_distinct_values = [0] * self.num_cols
        for j in range(self.num_cols):
            if self.is_col_categorical[j]:
                num_distinct_values[j] = self.get_num_distinct_values(j)
                num_cat_cols += 1

        # process pivots only if there are categorical columns
        if num_cat_cols > 0:
            if num_cat_cols == 1:  # single categorical column
                # find pivots
                j = 0
                # identify the categorical column j
                while not self.is_col_categorical[j]:
                    j += 1
                list_frequencies = sorted(self.maps_cat_cols[j].items(), key=lambda item: item[1])
                sorted_frequencies = [freq for _, freq in list_frequencies]
                # cut point for the MDL-based selection
                cut_point = self.minimum_description_length(sorted_frequencies)
                if cut_point == 0:
                    cut_point = num_distinct_values[j]

                # create pivots
                pivots = [None] * cut_point
                values = [0.0] * self.num_cols
                for k in range(cut_point):
                    values[j] = list_frequencies[k][0]
                    pivots[k] = Instance(k, values, 0)

                # update column
                for i in range(self.num_rows):
                    # compute and sum distances
                    value = sum(
                        self.instances[i].dist_cat(pivots[k], self.is_col_categorical, num_distinct_values) for k in
                        range(cut_point))
                    # set new value
                    self.instances[i].set_at(j, value)
            else:  # two or more categorical columns
                # find pivots
                ID_pivots = [0] * num_cat_cols
                random_index = random.randint(0, self.num_rows - 1)
                max_dist = self.instances[ID_pivots[0]].dist_cat(self.instances[random_index], self.is_col_categorical,
                                                                 num_distinct_values)
                for i in range(1, self.num_rows):
                    dist = self.instances[i].dist_cat(self.instances[random_index], self.is_col_categorical,
                                                      num_distinct_values)
                    if dist > max_dist:
                        max_dist = dist
                        ID_pivots[0] = i

                max_dist = self.instances[ID_pivots[1]].dist_cat(self.instances[ID_pivots[0]], self.is_col_categorical,
                                                                 num_distinct_values)
                for i in range(1, self.num_rows):
                    dist = self.instances[i].dist_cat(self.instances[ID_pivots[0]], self.is_col_categorical,
                                                      num_distinct_values)
                    if dist > max_dist:
                        max_dist = dist
                        ID_pivots[1] = i

                for next_pivot in range(2, num_cat_cols):
                    min_error = float('inf')
                    for i in range(self.num_rows):
                        error = 0
                        for k in range(next_pivot):
                            if i == ID_pivots[k]:
                                break
                            dist = self.instances[i].dist_cat(self.instances[ID_pivots[k]], self.is_col_categorical,
                                                              num_distinct_values)
                            error += abs(max_dist - dist)
                        if error < min_error:
                            min_error = error
                            ID_pivots[next_pivot] = i

                # copy pivots
                pivots = [self.instances[ID_pivots[k]] for k in range(num_cat_cols)]

                # update columns
                values = [0.0] * num_cat_cols
                for i in range(self.num_rows):
                    for k in range(num_cat_cols):
                        values[k] = self.instances[i].dist_cat(pivots[k], self.is_col_categorical, num_distinct_values)
                    j = -1
                    for k in range(num_cat_cols):
                        j += 1
                        while not self.is_col_categorical[j]:
                            j += 1
                        self.instances[i].set_at(j, values[k])

            # normalize updated column(s)
            for j in range(self.num_cols):
                if self.is_col_categorical[j]:
                    min_val = float('inf')
                    max_val = float('-inf')
                    for i in range(self.num_rows):
                        min_val = min(self.instances[i].get_at(j), min_val)
                        max_val = max(self.instances[i].get_at(j), max_val)
                    for i in range(self.num_rows):
                        value = (self.instances[i].get_at(j) - min_val) / (max_val - min_val)
                        self.instances[i].set_at(j, value)

    def is_normalized(self):
        return self.is_normalized

    def get_instance(self, i):
        return self.instances[i]

    def get_at(self, i, j):
        return self.instances[i].get_at(j)

    def as_list(self):
        return list(self.instances)

    def get_labels(self):
        return [self.get_instance(i).label for i in range(self.num_rows)]

    def get_num_distinct_values(self, j):
        assert self.maps_cat_cols[j] is not None
        return len(self.maps_cat_cols[j])

    def get_distinct_values(self, j):
        assert self.maps_cat_cols[j] is not None
        return list(self.maps_cat_cols[j].keys())

    def get_frequencies(self, j):
        assert self.maps_cat_cols[j] is not None
        return list(self.maps_cat_cols[j].values())

    def get_is_col_categorical(self):
        return self.is_col_categorical

    def has_label(self):
        return self.has_label

    def process_cat_cols(self):
        # get metadata about the categorical columns
        self.count_frequencies()
        num_cat_cols = 0
        num_distinct_values = [0] * self.num_cols
        for j in range(self.num_cols):
            if self.is_col_categorical[j]:
                num_distinct_values[j] = self.get_num_distinct_values(j)
                num_cat_cols += 1

        # process pivots only if there are categorical columns
        if num_cat_cols > 0:
            if num_cat_cols == 1:  # single categorical column
                # find pivots
                j = 0
                # identify the categorical column j
                while not self.is_col_categorical[j]:
                    j += 1
                list_frequencies = sorted(self.maps_cat_cols[j].items(), key=lambda item: item[1])
                sorted_frequencies = [freq for _, freq in list_frequencies]
                # cut point for the MDL-based selection
                cut_point = self.minimum_description_length(sorted_frequencies)
                if cut_point == 0:
                    cut_point = num_distinct_values[j]

                # create pivots
                pivots = [None] * cut_point
                values = [0.0] * self.num_cols
                for k in range(cut_point):
                    values[j] = list_frequencies[k][0]
                    pivots[k] = Instance(k, values, 0)

                # update column
                for i in range(self.num_rows):
                    # compute and sum distances
                    value = sum(
                        self.instances[i].dist_cat(pivots[k], self.is_col_categorical, num_distinct_values) for k in
                        range(cut_point))
                    # set new value
                    self.instances[i].set_at(j, value)
            else:  # two or more categorical columns
                # find pivots
                ID_pivots = [0] * num_cat_cols
                random_index = random.randint(0, self.num_rows - 1)
                max_dist = self.instances[ID_pivots[0]].dist_cat(self.instances[random_index], self.is_col_categorical,
                                                                 num_distinct_values)
                for i in range(1, self.num_rows):
                    dist = self.instances[i].dist_cat(self.instances[random_index], self.is_col_categorical,
                                                      num_distinct_values)
                    if dist > max_dist:
                        max_dist = dist
                        ID_pivots[0] = i

                max_dist = self.instances[ID_pivots[1]].dist_cat(self.instances[ID_pivots[0]], self.is_col_categorical,
                                                                 num_distinct_values)
                for i in range(1, self.num_rows):
                    dist = self.instances[i].dist_cat(self.instances[ID_pivots[0]], self.is_col_categorical,
                                                      num_distinct_values)
                    if dist > max_dist:
                        max_dist = dist
                        ID_pivots[1] = i

                for next_pivot in range(2, num_cat_cols):
                    min_error = float('inf')
                    for i in range(self.num_rows):
                        error = 0
                        for k in range(next_pivot):
                            if i == ID_pivots[k]:
                                break
                            dist = self.instances[i].dist_cat(self.instances[ID_pivots[k]], self.is_col_categorical,
                                                              num_distinct_values)
                            error += abs(max_dist - dist)
                        if error < min_error:
                            min_error = error
                            ID_pivots[next_pivot] = i

                # copy pivots
                pivots = [self.instances[ID_pivots[k]] for k in range(num_cat_cols)]

                # update columns
                values = [0.0] * num_cat_cols
                for i in range(self.num_rows):
                    for k in range(num_cat_cols):
                        values[k] = self.instances[i].dist_cat(pivots[k], self.is_col_categorical, num_distinct_values)
                    j = -1
                    for k in range(num_cat_cols):
                        j += 1
                        while not self.is_col_categorical[j]:
                            j += 1
                        self.instances[i].set_at(j, values[k])

            # normalize updated column(s)
            for j in range(self.num_cols):
                if self.is_col_categorical[j]:
                    min_val = float('inf')
                    max_val = float('-inf')
                    for i in range(self.num_rows):
                        min_val = min(self.instances[i].get_at(j), min_val)
                        max_val = max(self.instances[i].get_at(j), max_val)
                    for i in range(self.num_rows):
                        value = (self.instances[i].get_at(j) - min_val) / (max_val - min_val)
                        self.instances[i].set_at(j, value)

    def get_instance(self, i):
        return self.instances[i]

    def get_at(self, i, j):
        return self.instances[i].get_at(j)

    def as_list(self):
        return list(self.instances)

    def get_labels(self):
        return [self.get_instance(i).label for i in range(self.num_rows)]

    def get_num_rows(self):
        return self.num_rows

    def get_num_cols(self):
        return self.num_cols

    def get_name(self):
        return self.name

    def get_num_distinct_values(self, j):
        assert self.maps_cat_cols[j] is not None
        return len(self.maps_cat_cols[j])

    def get_distinct_values(self, j):
        assert self.maps_cat_cols[j] is not None
        return list(self.maps_cat_cols[j].keys())

    def get_frequencies(self, j):
        assert self.maps_cat_cols[j] is not None
        return list(self.maps_cat_cols[j].values())

    def get_is_col_categorical(self):
        return self.is_col_categorical

    def has_label(self):
        return self.has_label

    def save(self, path):
        datas = []
        for i in self.instances:
            row = []
            for index, v in enumerate(i.values):
                if self.is_col_categorical[index]:
                    row.append(v)
                else:
                    row.append(i.values_old[index])
            row.append('no' if i.label == 0 else ('yes' if i.label == 1 else i.label))
            datas.append(row)
        df = pd.DataFrame(datas, columns=self.header)
        df.to_csv(os.sep.join([path, self.get_name() + '.csv']), sep=',', index=False)


#path = r'C:\Users\pipip\Google Drive\Doutorado\Pesquisa\Experimentos\Categorial_Data\database\Ready\ADRepository\bank-additional-ful-nominal_v01.csv'
#relation = Relation.read_csv(path, ',', None, True)
#relation.normalize()
#relation.save(r'C:\Users\pipip\Google Drive\Doutorado\Pesquisa\Experimentos\Categorial_Data\database\Ready\ADRepository\number\pivot')
