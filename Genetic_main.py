"""
Train a Pong AI using Genetic algorithms.
"""


import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

def main():
    # init environment
    env = gym.make("Pong-v0")
    number_of_inputs = 3 # 3 actions: up, down, stay

    # init variables for genetic algorithms 
    generations = 20 # Number of times to evole the population.
    population = 50 # Number of networks in each generation.

    # init variables for CNN
    current_pool = []
    input_dim = (80,80,1)
    learning_rate = 0.001

    # Initialize all models
    for _ in range(population):
        """
        Keras 2.1.1; tensorflow as backend.


        Structure of CNN
        ----------------
        Convolutional Layer: 32 filers of 8 x 8 with stride 4 and applies ReLU activation function
            - output layer (width, height, depth): (20, 20, 32)

        MaxPooling Layer: 2 x 2 filers with stride 2
            - output layer (width, height, depth): (10, 10, 32)
        
        Dense Layer: fully-connected consisted of 32 rectifier units
            - output layer: 32 neurons

        Dropout Layer: 

        Output Layer: fully-connected linear layer with a single output for each valid action, applies softmax activation function
        

        Refernce: https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py

        """
        model = Sequential()
        model.add(Reshape((80,80,1), input_shape=(input_dim,)))
        model.add(Conv2D(32, kernel_size = (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        current_pool.append(model)
    
    def preprocessImage(I):
        """ Return array of 80 x 80
        https://github.com/mkturkcan/Keras-Pong/blob/master/keras_pong.py
        """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I


    




if __name__ == '__main__':
    main()
    
