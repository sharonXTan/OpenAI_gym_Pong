# OpenAI_gym_Pong

#### Environment:

In the game of Pong, the policy could take the pixels of the screen and compute the probability of moving the player’s paddle Up, Down, or neither. (https://blog.openai.com/evolution-strategies/)


- Initial inout pixel: 210x160x3 [width 210, height 160, and with three color channels R,G,B.]

Steps for processing Image before feeding to CNN (from https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0)

1. Crop the image (we just care about the parts with information we care about).
2. Downsample the image.
3. Convert the image to black and white (color is not particularly important to us).
4. Remove the background.
5. (Maybe not suitiable for CNN)Convert from an 80 x 80 matrix of values to 6400 x 1 matrix (flatten the matrix so it’s easier to use). 
6. Store just the difference between the current frame and the previous frame if we know the previous frame (we only care about what’s changed).


