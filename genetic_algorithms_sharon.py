"""
This hold the logic of genetic algorithms.

Reference: 
https://github.com/erilyth/Flappy-Bird-Genetic-Algorithms


"""

import numpy as np

def crossover(current_generation,model_idx1, model_idx2):
    """
    Crossover two neural network to produce two new networks by 
    swapping the weights randomly (layer 1, 4, and 6; Conv2d, dense ,output)


    Attributes:
        current_generation(list): A list of models at current generation
        model_idx1(int): Index of first model
        model_idx2(int): Index of secound model

    """
    weights1 = current_generation[model_idx1].get_weights()
    weights2 = current_generation[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    swap_layer = np.random.choice([1,4,6])
    weightsnew1[swap_layer] = weights2[swap_layer]
    weightsnew2[swap_layer] = weights1[swap_layer]
    return np.asarray([weightsnew1, weightsnew2])


def mutate(weights):
    """Select weights randomly with a 0.15 probability and then change its value with a random number between -0.5 to +0.5.
    """
    for xi in range(len(weights)):
	    for yi in range(len(weights[xi])):
            if np.random.uniform(0, 1) > 0.85:
			    change = np.random.uniform(-0.5,0.5)
				weights[xi][yi] += change
	return weights

def fitness(reward_sum)
    observation = env.reset()
	action = env.action_space.sample()
    running_reward = None
    reward_sum = 0
    episode_number = 0
	drs = []
    while True:
        if render: 
	         env.render()
    # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward 
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        if done: # an episode finished
             episode_number += 1
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    reward_sum -= np.mean(reward_sum)
    reward_sum /= np.std(reward_sum)
	# define fitness condition
	fitness(reward_sum) = np.maximum(reward_sum)
	
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epr = np.vstack(drs)
    drs = [] # reset array memory

"""
    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
"""