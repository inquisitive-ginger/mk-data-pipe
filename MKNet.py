import time
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import math
from threading import Thread
from MKModel import FinalNet, OriginalNet

class MKNet(object):
    def __init__(self, serial_instance, action_server_instance, env_instance, params, results):
        self.ctx = mx.cpu()

        #  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # params for gluon trainer
        self.gamma = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.0000000001
        self.momentum_param = 0.05
        self.learning_rates = [0.0001, 0.01]

        self.actions = []
        # serial initialization
        self.mk_serial = serial_instance
        self.env = env_instance
        self.action_server = action_server_instance
        self.result_writer = results

        self.num_action = len(self.env.action_space)

        self.params = params
        self.loss = gluon.loss.L2Loss()
        self.model = OriginalNet(self.num_action)
        
        # initialize random params or from file
        if(self.params is not None):
            self.model.load_params(self.params, ctx=self.ctx)
        else:
            self.model.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)           
        
        self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': self.learning_rate,  "beta1": self.beta1,  "beta2": self.beta2, "epsilon": self.epsilon})        
        # self.action_optimizer = gluon.Trainer(self.model.value_pred.collect_params(), 'adam', {'learning_rate': self.learning_rate,  "beta1": self.beta1,  "beta2": self.beta2, "epsilon": self.epsilon})
        # self.value_optimizer = gluon.Trainer(self.model.action_pred.collect_params(), 'adam', {'learning_rate': self.learning_rate,  "beta1": self.beta1,  "beta2": self.beta2, "epsilon": self.epsilon})
    # start a sperate training thread that will push actions into a list
    # made available to an action server
    def play(self):
        train_thread = Thread(target=self.train)
        train_thread.start()
        self.action_server.run_command_server(self.actions)

    # step through the environment and take actions based on policy and value
    # function approximations
    def train(self):
        print("Starting training...")
        train_scores = [0]

        for episode in range(0, self.env.episodes):
            # modify this line below env.reset should send back the next pack of 8 frames
            # we could use instead of env.reset the preprocess function
            self.action_server.reset_last_action()
            next_frame_bundle = self.env.reset()
            s1 = next_frame_bundle

            # update the number of steps depending on number of episodes
            if(episode % 100 == 0 and episode != 0):
                self.env.learning_steps += 10

            rewards = []
            values = []
            actions = []
            heads = []

            with autograd.record():
                for learning_step in range(self.env.learning_steps):
                    # Converts and down-samples the input image
                    before_model = time.time()
                    prob, value = self.model(s1)
                    after_model = time.time()
                    # print("MODEL TIME: {}".format(after_model - before_model))
                    # dont always take the argmax, instead pick randomly based on probability
                    index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
                    action = index.asnumpy()[0].astype(np.int64)
                    # self.actions.append(self.env.action_map[action])
                    self.actions.append(action)

                    # take one step in the envrionment
                    start_step = time.time()
                    next_frame_bundle, rew, done = self.env.step(action, learning_step)
                    stop_step = time.time()

                    print("EP: {:<5} | STEP {:<3} | ACTION: {:<12} | REWARD: {:4f}".format(episode, learning_step, self.env.action_space[action], rew))
                    # print("STEP TIME: {}".format(stop_step - start_step))

                    isterminal = done
                    rewards.append(rew)
                    actions.append(action)
                    values.append(value)
                    heads.append(logp)

                if isterminal:
                    #print("finished_game")
                    break

                s1 = next_frame_bundle if not isterminal else None
                # reverse accumulate and normalize rewards
                R = 0
                for i in range(len(rewards) - 1, -1, -1):
                    R = rewards[i] + self.gamma * R
                    rewards[i] = R
                rewards = np.array(rewards)
                train_scores.append(np.sum(rewards))
                # print("Rewards: {}".format(rewards))
                print("Episode Reward: {}".format(np.sum(rewards)))
                #rewards -= rewards.mean()
                #rewards /= rewards.std() + np.finfo(rewards.dtype).eps

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
            # self.action_optimizer.step(s1.shape[0])
            # self.value_optimizer.step(s1.shape[0])

            train_scores = np.array(train_scores)
            self.result_writer.append_results("{},{:2},{:2},{:2}\n".format(episode, train_scores.mean(), train_scores.min(), train_scores.max()))
            train_scores = train_scores.tolist()

            if episode % self.env.display_count == 0 and episode != 0:
                train_scores = np.array(train_scores)            
                print("Episodes {}\t".format(episode),
                    "Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                    "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max(),
                    "actions: ", np.unique(actions, return_counts=True))
                train_scores = []
            if episode % 5 == 0 and episode != 0:
                self.model.save_params("./params/mkEpisodes_%d.params" % episode)
                pass