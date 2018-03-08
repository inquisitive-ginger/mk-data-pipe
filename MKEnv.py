import time
from MKDirectionDetect import MKDirectionDetect
import mxnet as mx

class MKEnv(object):
    def __init__(self, capture_instance, mk_serial, num_episodes, learning_steps, display_count):
        self.episodes = num_episodes  # Number of episodes to be played
        self.learning_steps = learning_steps  # Maximum number of learning steps within each episodes
        self.display_count = display_count  # The number of episodes to play before showing statistics.
        
        self.capture = capture_instance
        # self.action_space = ['accel', 'brake', 'left_accel', 'right_accel']
        self.action_space = ['accel', 'left_accel', 'right_accel']
        self.mk_serial = mk_serial
        self.reset_commands = ['-', 's', 'j']
        # Loading model to predict direction of Mario
        self.directionModel = MKDirectionDetect(ctx=mx.cpu())
        self.directionModel.dirNet.load_params("./model.params", ctx=mx.cpu())

    def reset(self):
        for command in self.reset_commands:
            self.mk_serial.send_command(command)
            time.sleep(1)
        print('Resetting environment...')
        time.sleep(10)
        print('Let\'s do this thing!!!')
        self.capture.set_frame_bundle()
        return self.capture.get_transposed_frames()

    # def calc_reward(self, of):
    #     reward = 0
    #     if(of > 5):
    #         reward = of
    #     else:
    #         reward = -1
    #     return reward
    
    def step(self, action):
        # last_frame used to detect the direction
        last_frame = self.capture.set_frame_bundle()
        # print('last frame:', last_frame)    
        print("Time @ Bundle: ")
        transposed_frames = self.capture.get_transposed_frames()
        
        direction = self.directionModel.classify_direction(last_frame)

        if direction == 'backward':
            reward = -1000
        else: 
            optical_flow = self.capture.calc_optical_flow()
            reward = 1/optical_flow
            # reward = self.calc_reward(optical_flow)

        return transposed_frames, reward, False