# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:44:40 2022

@author: braha
"""

import tensorflow as tf 
import numpy as np


# create causal and dilated convolution 
class CasualDilatedConv1D():
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super().__init__()
        self.conv1D =tf.keras.layers.Conv1D(in_channels, out_channels, kernel_size,padding='valid',dilation_rate=1)
        print("81")
        #self.conv1D=tf.keras.layers.Conv1D(in_channels, out_channels,filters=1, kernel_size,padding='valid',dilation_rate=1)
        self.ignoreOutIndex = (kernel_size - 1) * dilation_rate


    def forward(self, x):
        return self.conv1D(x)[..., :-self.ignoreOutIndex]


class DenseLayer():
    def __init__(self, in_channels):
        super().__init__()
        self.relu = tf.keras.layers.ReLU
        self.softmax = tf.keras.layers.Softmax()
        self.kernel_size=1
        self.conv1d = tf.keras.layers.Conv1D(in_channels, in_channels,self.kernel_size )

    def forward(self, skipConnection):
        # as b c outputsize -> skipConnection size
        out = tf.keras.layers.mean(skipConnection, dim=0)

        for i in range(2):
            out = self.relu(out)
            out = self.conv1d(out)
        return self.softmax(out)


class ResBlock():
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.casualDilatedConv1D = CasualDilatedConv1D(res_channels, res_channels, kernel_size, dilation)
        #self.resConv1D = tf.keras.layer.Conv1D(res_channels, res_channels, kernel_size=1)
        self.resConv1D=tf.keras.layers.Conv1D(res_channels, res_channels, kernel_size,padding='valid',dilation_rate=1)
        #self.skipConv1D = tf.keras.layer.Conv1D(res_channels, skip_channels, kernel_size)
        self.skipConv1D =tf.keras.layers.Conv1D(res_channels, skip_channels, kernel_size,padding='valid',dilation_rate=1)
        self.tanh = tf.keras.activations.tanh
        self.sigmoid = tf.keras.activations.sigmoid

    def forward(self, inputX, skipSize):
        x = self.casualDilatedConv1D(inputX)
        x1 = self.tanh(x)
        x2 = self.sigmoid(x)
        x = x1 * x2
        resOutput = self.resConv1D(x)
        resOutput = resOutput + inputX[..., -resOutput.size(2):]
        skipOutput = self.skipConv1D(x)
        skipOutput = skipOutput[..., -skipSize:]
        return resOutput, skipOutput


class StackOfResBlocks():

    def __init__(self, stack_size, layer_size, res_channels, skip_channels, kernel_size):
        super().__init__()
        buildDilationFunc = np.vectorize(self.buildDilation)
        dilations = buildDilationFunc(stack_size, layer_size)
        self.resBlocks = []
        for s,dilationPerStack in enumerate(dilations):
            for l,dilation in enumerate(dilationPerStack):
                resBlock=ResBlock(res_channels, skip_channels, kernel_size, dilation)
                #self.add_module(f'resBlock_{s}_{l}', resBlock) # Add modules manually
                self.resBlocks.append(resBlock)

    def buildDilation(self, stack_size, layer_size):
        # stack1=[1,2,4,8,16,...512]
        dilationsForAllStacks = []
        for stack in range(stack_size):
            dilations = []
            for layer in range(layer_size):
                dilations.append(2 ** layer)
            dilationsForAllStacks.append(dilations)
        return dilationsForAllStacks

    def forward(self, x, skipSize):
        resOutput = x
        skipOutputs = []
        for resBlock in self.resBlocks:
            resOutput, skipOutput = resBlock(resOutput, skipSize)
            skipOutputs.append(skipOutput)
        return resOutput, tf.keras.layers.stack(skipOutputs)


class WaveNet():
    def __init__(self, in_channels, out_channels, kernel_size, stack_size, layer_size):
        super().__init__()
        self.stack_size = stack_size
        self.layer_size = layer_size
        self.kernel_size = kernel_size
        print("60")
        self.casualConv1D = CasualDilatedConv1D(in_channels, out_channels, kernel_size,1)
        self.stackResBlock = StackOfResBlocks(self.stack_size, self.layer_size, in_channels, out_channels, kernel_size)
        self.denseLayer = DenseLayer(out_channels)


    def calculateReceptiveField(self):
        return np.sum([(self.kernel_size - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)

    def calculateOutputSize(self, x):
        return int(x.size(2)) - self.calculateReceptiveField()

    def forward(self, x):
        # x: b c t -> input data size
        x = self.casualConv1D(x)
        skipSize = self.calculateOutputSize(x)
        _, skipConnections = self.stackResBlock(x, skipSize)
        dense=self.denseLayer(skipConnections)
        return dense
    
class WaveNetClassifier():
    def __init__(self,seqLen,output_size):
        super().__init__()
        self.output_size=output_size
        self.wavenet=WaveNet(1,1,2,3,4)
        self.liner=tf.keras.layers.Dense(seqLen-self.wavenet.calculateReceptiveField(),output_size)
        self.softmax=tf.keras.layers.Softmax(-1)
    
    def forward(self,x):
        x=self.wavenet(x)
        x=self.liner(x)
        return self.softmax(x)
        