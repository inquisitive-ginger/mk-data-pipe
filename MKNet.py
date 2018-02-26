import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import os
import random
import numpy as np
import math
import time
from time import sleep
import mxnet.ndarray as F
from threading import Thread

from MKEnv import MKEnv
from ActionServer import ActionServer

# game environment init
EPISODES = 1000000  # Number of episodes to be played
LEARNING_STEPS = 600  # Maximum number of learning steps within each episodes
DISPLAY_COUNT = 10  # The number of episodes to play before showing statistics.

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

class MKNet(object):
    def __init__(self):
        # ctx = mx.gpu()
        self.ctx = mx.cpu()

        #  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # params for gluon trainer
        self.gamma = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.0001
        self.momentum_param = 0.05
        self.learning_rates = [0.0001, 0.01]

        self.env = MKEnv(bundle_size=8)
        self.actions = []
        self.frame_repeat = 4
        self.num_action = len(self.env.action_space)
        self.action_server = ActionServer('ws://192.168.4.1:80/ws', self.actions)

        self.loss = gluon.loss.L2Loss()
        self.model = Net(self.num_action)
        self.model.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': self.learning_rate,  "beta1": self.beta1,  "beta2": self.beta2, "epsilon": self.epsilon})

    # start a sperate training thread that will push actions into a list
    # made available to an action server
    def play(self):
        train_thread = Thread(target=self.train)
        train_thread.start()
        self.action_server.run_command_server()

    # step through the environment and take actions based on policy and value
    # function approximations
    def train(self):
        print("Training has begun....")
        episode_rewards = 0
        final_rewards = 0

        running_reward = 10 
        train_episodes_finished = 0
        train_scores = [0]
        num_action_index = 0

        for episode in range(0, EPISODES):
            # modify this line below env.reset should send back the next pack of 8 frames
            # we could use instead of env.reset the preprocess function
            next_frame_bundle = self.env.reset()
            s1 = next_frame_bundle

            rewards = []
            values = []
            actions = []
            heads = []

            with autograd.record():
                for learning_step in range(LEARNING_STEPS):
                    # Converts and down-samples the input image
                    prob, value = self.model(s1)
                    # dont always take the argmax, instead pick randomly based on probability
                    index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
                    action = index.asnumpy()[0].astype(np.int64)
                    self.actions.append(self.env.action_map[action])
                    
                    # print('#', num_action_index,': ' , 'action Number: ', action, self.env.action_space[action])
                    num_action_index += 1

                    # skip frames
                    reward = 0
                    # env step could be a set of funtions:
                        # a function that packages 8 frames
                        # a function that sends back the optical flow
                        # when these two functions returns something we can set done (below) to true
                        # not sure about the underscore
                    next_frame_bundle, rew, done = self.env.step(action)

                    reward += rew
                    print("EP: {:<5} | STEP {:<3} | ACTION: {:<8} | REWARD: {:4f}".format(episode, learning_step, self.env.action_space[action], rew))

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
                    R = rewards[i] + self.gamma * R
                    rewards[i] = R
                rewards = np.array(rewards)
                rewards -= rewards.mean()
                rewards /= rewards.std() + np.finfo(rewards.dtype).eps

                # compute loss and gradient
                L = sum([self.loss(value, mx.nd.array([r]).as_in_context(self.ctx)) for r, value in zip(rewards, values)])
                final_nodes = [L]
                for logp, r, v in zip(heads, rewards, values):
                    reward = r - v.asnumpy()[0, 0]
                    # Here we differentiate the stochastic graph, corresponds to the
                    # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
                    # Optimizer minimizes the loss but we want to maximizing the reward,
                    # so use we use -reward here.
                    final_nodes.append(logp * (-reward))
                autograd.backward(final_nodes)
            self.optimizer.step(s1.shape[0])

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