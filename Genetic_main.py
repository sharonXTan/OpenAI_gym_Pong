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
    input_dim = (80,80,1)
    learning_rate = 0.001

    # Initialize all models
    for i in range(population):
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

        Dense Layer: fully-connected linear layer with a single output for each valid action, applies softmax activation function


        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_uniform',
        input_shape=(80,80,1)))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt)





if __name__ == '__main__':
    main()
    
