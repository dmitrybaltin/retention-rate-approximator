#@title Import libraries, define classes and functions
import numpy as np
import torch
import torch.nn as nn

import os
from google.colab import files
import pandas as pd
import io
import matplotlib.pyplot as plt

class ConstantFunction(torch.nn.Module):
    def __init__(self, initial_weights=None):

        super(ConstantFunction, self).__init__()

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor([initial_weights]))
        else:
            self.w = nn.Parameter(torch.Tensor([0.25]))

    def forward(self, x):
        ret_value = torch.ones_like(x)*self.w[0]
        return ret_value

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0]
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):
        new_w = torch.mean(y_train).unsqueeze(0)
        self.w = nn.Parameter(new_w)

class LinearFunction(torch.nn.Module):
    def __init__(self, initial_weights=None):

        super(LinearFunction, self).__init__()

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.Tensor([0.25, 0.25]))

    def forward(self, x):
        return x * self.w[1] + self.w[0]

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0, 0]
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):
        firstValue = y_train[torch.argmin(x_train)]
        lastValue =  y_train[torch.argmax(x_train)]

        new_w = [lastValue, firstValue - lastValue]
        self.w = nn.Parameter(torch.Tensor(new_w))

# Approximation of a decreasing process by a linear-fractional function
class InverseFunction(torch.nn.Module):
    def __init__(self, initial_weights=None):

        super(InverseFunction, self).__init__()

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.Tensor([0.25, 0.25, 1]))

    def forward(self, x):
        return self.w[0] + torch.ones_like(x) * self.w[1] / (x + self.w[2])

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0, 0, 10]  # The value 10 was obtained by experience
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):  # The value 1 was obtained by experience

        firstValue = y_train[torch.argmin(x_train)]
        lastValue =  y_train[torch.argmax(x_train)]

        new_w = [lastValue, firstValue - lastValue, 1.0]
        self.w = nn.Parameter(torch.Tensor(new_w))

class InverseFunction_4w(torch.nn.Module):
    def __init__(self, initial_weights=None):

        super(InverseFunction_4w, self).__init__()

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.Tensor([0.25, 0.25, 1, 1]))

        # todo: !!!! add constrains. all the weights must be > 0

    def forward(self, x):

        return self.w[0] + torch.ones_like(x) * self.w[1] / (torch.pow(x, self.w[3]) + self.w[2])

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0, 0, 10, 1]  # The value 10 was obtained by experience
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):  # The value 1 was obtained by experience

        firstValue = y_train[torch.argmin(x_train)]
        lastValue =  y_train[torch.argmax(x_train)]

        new_w = [lastValue, firstValue - lastValue, 1.0, 1.0]
        self.w = nn.Parameter(torch.Tensor(new_w))

class LinearFractionalFunction_new(torch.nn.Module):
    def __init__(self, initial_weights=None):

        super(LinearFractionalFunction_new, self).__init__()

        #print('LinearFractionalDecreaser3w_v2 weights = {0}'.format(initial_weights))

        if initial_weights is not None:
            self.w0 = nn.Parameter(torch.Tensor([initial_weights[0]]))
            self.w1 = nn.Parameter(torch.Tensor([initial_weights[1]]))
            self.w2 = nn.Parameter(torch.Tensor([initial_weights[2]]))
        else:
            self.w0 = nn.Parameter(torch.Tensor([0.5]))
            self.w1 = nn.Parameter(torch.Tensor([0.25]))
            self.w2 = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x):

        return (x / (x + 1/self.w2[0])) * (self.w1[0] - self.w0[0]) + self.w0[0]
        #return self.w[0] - (x / (x + torch.pow(self.w[2],2))) * self.w[1]

    def reset_weights(self, new_w=None):
        if new_w is None:
            self.w0 = nn.Parameter(torch.Tensor([0.5]))
            self.w1 = nn.Parameter(torch.Tensor([0.25]))
            self.w2 = nn.Parameter(torch.Tensor([0.1]))
        else:
            self.w0 = nn.Parameter(torch.Tensor([new_w[0]]))
            self.w1 = nn.Parameter(torch.Tensor([new_w[1]]))
            self.w2 = nn.Parameter(torch.Tensor([new_w[2]]))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):  # The value 10 was obtained by experience

        firstValue = torch.max(y_train)
        lastValue =  torch.min(y_train)
        w0 = firstValue
        w1 = firstValue - lastValue

        if w0 < 0:
            w0 = 0.1
        if w1 < 0:
            w1 = 0
        if w1 > w0:
            w1 = w0

        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)

        if x_mean != 0:
            w2 = ( w1 / (w0 - y_mean) - 1 ) / x_mean
        else:
            w2 = 0.05

        self.reset_weights([w0, w1, w2])

class LinearFractionalFunction(torch.nn.Module):
    def __init__(self, initial_weights=None):

        #super(LinearFractionalFunction, self).__init__()
        super().__init__()

        #print('LinearFractionalDecreaser3w_v2 weights = {0}'.format(initial_weights))

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 20]))

    def forward(self, x):

        return (x / (x + self.w[2])) * (-self.w[1]) + self.w[0]

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0, 0, 10]  # The value 10 was obtained by experience
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):  # The value 10 was obtained by experience

        firstValue = torch.max(y_train)
        lastValue =  torch.min(y_train)
        w0 = firstValue
        w1 = firstValue - lastValue

        if w0 < 0:
            w0 = 0.1
        if w1 < 0:
            w1 = 0
        if w1 > w0:
            w1 = w0

        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)

        if y_mean - w0 != 0:
            w2 = x_mean * ( w1 / (w0 - y_mean) - 1 )
        else:
            w2 = 20

        self.reset_weights([w0, w1, w2])

class SigmaFunction(torch.nn.Module):
    def __init__(self, initial_weights=None):

        #super(LinearFractionalFunction, self).__init__()
        super().__init__()

        #print('LinearFractionalDecreaser3w_v2 weights = {0}'.format(initial_weights))

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 20]))

    def forward(self, x):

        return torch.nn.Sigmoid()(-x*self.w[2]*2) * self.w[1] + self.w[0]

    def reset_weights(self, new_w=None):
        if new_w is None:
            new_w = [0, 0, 10]  # The value 10 was obtained by experience
        self.w = nn.Parameter(torch.Tensor(new_w))

    # Initialize weights using train dataset
    def init_weights_from_train_data(self, x_train, y_train):  # The value 10 was obtained by experience

        firstValue = torch.max(y_train)
        lastValue =  torch.min(y_train)
        w0 = firstValue
        w1 = firstValue - lastValue

        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)

        if x_mean != 0:
            w2 = ( w1 / (w0 - y_mean) - 1 ) / x_mean
        else:
            w2 = 0.05

        self.reset_weights([w0, w1, w2])

# Approximation of weekly fluctuations. Every day of week has its own constant weight
class WeekFunction(torch.nn.Module):
    def __init__(self, first_day_of_week:int, regularizer_base:int, initial_weights=None):
        super(WeekFunction, self).__init__()

        self.first_day_of_week = first_day_of_week

        if initial_weights is not None:
            self.w = nn.Parameter(torch.Tensor(initial_weights))
        else:
            self.w = nn.Parameter(torch.ones(7))

        self.regularizer_base = regularizer_base

    def forward(self, day_numbers):
        day_of_week = (day_numbers.type(torch.long) + 7 - self.first_day_of_week) % 7
        weight_values = self.w[day_of_week]
        return weight_values

    def regularize(self):
        return torch.square(torch.mean(self.w) - self.regularizer_base)


def multiply_connector(input1, input2):
    return torch.mul(input1, input2)

def additive_connector(input1, input2):
    return torch.add(input1, input2)


class ApproximatorsFactory():

    main_functions = [
        ('w0', ConstantFunction, [0.25]),
        ('w0+w1*x', LinearFunction, [0.25, 0]),
        ('w0+w1/(w2+x)', InverseFunction, [0.25, 0.25, 1.0]),
        ('w0-w1*x/(w2+x)', LinearFractionalFunction, [0.5, 0.25, 10.0]),
        ('w0-(w0-w1)*x/(1/w2+x)', LinearFractionalFunction_new, [0.5, 0.25, 0.1]),
        ('w0+w1/(w2+pow(x,w3))', InverseFunction_4w, [0.25, 0.25, 1.0, 1]),
        ('w0+w1*Sigmoid(x*w3)', SigmaFunction, [0.5, 0.25, 0.05])]

    chain_functions = [
        ('w0', ConstantFunction, [0.25]),
        ('w0+w1*x', LinearFunction, [0.25, 0])]

    connectors = [
        ('mul', multiply_connector,   1),
        ('add', additive_connector,   0)]

    @staticmethod
    def create_main_function(function_type, initial_weights):

        if function_type is None:
            return ApproximatorsFactory.main_functions[0][1](initial_weights)

        for index, row in enumerate(ApproximatorsFactory.main_functions):
            if function_type == index or function_type == str(index) or function_type == row[0]:
                return row[1](initial_weights)

    @staticmethod
    def create_chain_function(function_type, initial_weights):

        if function_type is None:
            return ApproximatorsFactory.chain_functions[0][1](initial_weights)

        for index, row in enumerate(ApproximatorsFactory.chain_functions):
            if function_type == index or function_type == str(index) or function_type == row[0]:
                return row[1](initial_weights)

        return ApproximatorsFactory.chain_functions[0][1](initial_weights)

    @staticmethod
    def create_connector(connector_type):

        if connector_type is None:
            row = ApproximatorsFactory.connectors[0]
            return  row[1], row[2]

        for index, row in enumerate(ApproximatorsFactory.connectors):
            if connector_type == index or connector_type == str(index) or connector_type == row[0]:
                return row[1], row[2]

        row = ApproximatorsFactory.connectors[0]
        return  row[1], row[2]

    @staticmethod
    def get_main_function_weights_number(function_type):
        if function_type is None:
            return None

        for index, row in enumerate(ApproximatorsFactory.main_functions):
            if function_type == index or function_type == str(index) or function_type == row[0]:
                return row[2]

        return None