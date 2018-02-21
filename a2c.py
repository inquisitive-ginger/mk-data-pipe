import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import os
import random
import numpy as np
# from IPython import display
# import gym
import math
import time
from time import sleep
import mxnet.ndarray as F
import itertools as it

from MKEnv import MKEnv

# import matplotlib.pyplot as plt
# %matplotlib inline

# !sudo apt-get install python3-pip
# !pip3 install gym[atari]


# game environment init
EPISODES = 1000000  # Number of episodes to be played
LEARNING_STEPS = 600  # Maximum number of learning steps within each episodes
DISPLAY_COUNT = 10  # The number of episodes to play before showing statistics.

#  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# params for gluon trainer
gamma = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.0001
momentum_param = 0.05
learning_rates = [0.0001, 0.01]

# Other parameters
frame_repeat = 4

# ctx = mx.gpu()
ctx = mx.cpu()

# env_name = 'AssaultNoFrameskip-v4' # Set the desired environment
# env_name = "AsteroidsNoFrameskip-v0"
# env = gym.make(env_name)

# num_action = env.action_space.n # Extract the number of available action from the environment setting
env = MKEnv(bundle_size=8)
num_action = len(env.action_space)

# gluon.Block is the basic building block of models.
# You can define networks by composing and inheriting Block:
class Net(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(16, kernel_size=5, strides=2)
            self.bn1 = gluon.nn.BatchNorm()
            self.conv2 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn2 = gluon.nn.BatchNorm()
            self.conv3 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn3 = gluon.nn.BatchNorm()
            #self.lstm = gluon.rnn.LSTMCell(128)
            self.dense1 = gluon.nn.Dense(128, activation='relu')
            self.dense2 = gluon.nn.Dense(64, activation='relu')
            self.action_pred = gluon.nn.Dense(available_actions_count)
            self.value_pred = gluon.nn.Dense(1)
        #self.states = self.lstm.begin_state(batch_size=1, ctx=ctx)

    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = nd.flatten(x).expand_dims(0)
        #x, self.states = self.lstm(x, self.states)
        x = self.dense1(x)
        x = self.dense2(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values

loss = gluon.loss.L2Loss()
model = Net(num_action)
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate,  "beta1": beta1,  "beta2": beta2, "epsilon": epsilon})

# def preprocess(raw_frame):
#     # raw_frame is an array of frames
#     raw_frame = nd.array(raw_frame,mx.cpu())
#     #raw_frame = nd.reshape(nd.mean(raw_frame, axis = 2),shape = (raw_frame.shape[0],raw_frame.shape[1],1))
#     raw_frame = mx.image.imresize(raw_frame,  84, 84)
#     raw_frame = nd.transpose(raw_frame, (2,0,1))
#     raw_frame = raw_frame.astype(np.float32)/255.
#     data = nd.array(raw_frame).as_in_context(ctx)
#     data = data.expand_dims(0)
#     return data

render_image = False

def train():
    print("Start the training!")
    episode_rewards = 0
    final_rewards = 0

    running_reward = 10 
    train_episodes_finished = 0
    train_scores = [0]
    num_action_index = 0

    for episode in range(0, EPISODES):
        # modify this line below env.reset should send back the next pack of 8 frames
        # we could use instead of env.reset the preprocess function
        next_frame_bundle = env.reset()
        # proper_frame = next_frame
        # s1 = preprocess(proper_frame)
        next_frame_bundle = next_frame_bundle
        s1 = next_frame_bundle

        rewards = []
        values = []
        actions = []
        heads = []

        with autograd.record():
            for learning_step in range(LEARNING_STEPS):
                # Converts and down-samples the input image
                prob, value = model(s1)
                # dont always take the argmax, instead pick randomly based on probability
                index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
                action = index.asnumpy()[0].astype(np.int64)
                
                print('#', num_action_index,': ' , 'action Number: ', action, env.action_space[action])
                num_action_index += 1

                # skip frames
                reward = 0
                for skip in range(frame_repeat+1):
                    # do some frame math to make it not all jumpy and weird
                    # env step could be a set of funtions:
                        # a function that packages 8 frames
                        # a function that sends back the optical flow
                        # when these two functions returns something we can set done (below) to true
                        # not sure about the underscore
                    # new_next_frame, rew, done, _ = env.step(action)
                    next_frame_bundle, rew, done = env.step(action)
                    print('reward:', reward)
                    # proper_frame = next_frame + new_next_frame 
                    # next_frame = new_next_frame

                    # can render image if we want
                    #renderimage(proper_frame)
                    reward += rew
                #reward = game.make_action(doom_actions[action], frame_repeat)

                isterminal = done
                rewards.append(reward)
                actions.append(action)
                values.append(value)
                heads.append(logp)

                if isterminal:       
                    #print("finished_game")
                    break
                s1 = next_frame_bundle if not isterminal else None
            train_scores.append(np.sum(rewards))
            # reverse accumulate and normalize rewards
            R = 0
            for i in range(len(rewards) - 1, -1, -1):
                R = rewards[i] + gamma * R
                rewards[i] = R
            rewards = np.array(rewards)
            rewards -= rewards.mean()
            rewards /= rewards.std() + np.finfo(rewards.dtype).eps

            # compute loss and gradient
            L = sum([loss(value, mx.nd.array([r]).as_in_context(ctx)) for r, value in zip(rewards, values)])
            final_nodes = [L]
            for logp, r, v in zip(heads, rewards, values):
                reward = r - v.asnumpy()[0, 0]
                # Here we differentiate the stochastic graph, corresponds to the
                # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
                # Optimizer minimizes the loss but we want to maximizing the reward,
                # so use we use -reward here.
                final_nodes.append(logp * (-reward))
            autograd.backward(final_nodes)
        optimizer.step(s1.shape[0])

        if episode % DISPLAY_COUNT == 0:
            train_scores = np.array(train_scores)
            print("Episodes {}\t".format(episode),
                  "Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max(),
                  "actions: ", np.unique(actions, return_counts=True))
            train_scores = []
        if episode % 1000 == 0 and episode != 0:
            # model.save_params("/data/asteroids.params")
            pass
train()

# model.load_params("/data/asteroids.params")

# def renderimage(next_frame):
#     plt.imshow(next_frame)
#     plt.show()
#     display.clear_output(wait=True)
#     time.sleep(.01)

# def run_episode():
#     next_frame = env.reset()
#     done = False
#     while not done:
#         s1 = preprocess(next_frame)
#         prob, value = model(s1)
#         index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
#         action = index.asnumpy()[0].astype(np.int64)
#         new_next_frame, rew, done, _ = env.step(action)
#         proper_frame = next_frame + new_next_frame 
#         next_frame = new_next_frame
#         renderimage(proper_frame)
#         next_frame = new_next_frame

# run_episode()