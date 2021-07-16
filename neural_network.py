import tensorflow as tf
import pandas as pd
import numpy as np
import scipy 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function 
passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer 
(only applicable if use_bias is True). These are all attributes of Dense.
'''
#bias_initializer0=tf.keras.initializers.Zeros();
#bias_initializer0=tf.keras.initializers.random_normal() https://keras.io/api/layers/initializers/ 
bias_initializer0=tf.keras.initializers.Constant(value=0.4)
matA = np.linspace(-10,37,48).reshape(3,16)
dns1 = tf.keras.layers.Dense(units=2, activation='linear', use_bias=True, 
                             bias_initializer=bias_initializer0 )

print('dns1, activation, name=', dns1.activation.__name__)
dns2=tf.keras.layers.Dense(units=5, activation='linear', use_bias=True, bias_initializer=bias_initializer0)
dns2(dns1(matA))
#
# print('dns1 weights=', dns1.get_weights() )
check_val=dns2(dns1(matA)) - ( np.matmul(np.matmul( matA, dns1.get_weights()[0] ) + 0.4, dns2.get_weights()[0]) + 0.4)

print('check=\n ', check_val)