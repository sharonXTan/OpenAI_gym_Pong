"""
Train a Pong AI using Genetic algorithms.
"""


import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

def main():
    # init environment
    env = gym.make("Pong-v0")
    number_of_inputs = 3 # 3 actions: up, down, stay

    # init variables for genetic algorithms 
    generations = 20 # Number of times to evole the population.
    population = 50 # Number of networks in each generation.

    # init variables for CNN
    input_dim = 80 * 80 
    learning_rate = 0.001

    # Initialize all models
    for i in range(population):
        model = Sequential()
        model.add(Reshape((1,80,80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu', init='he_uniform'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)





if __name__ == '__main__':
    main()
    
