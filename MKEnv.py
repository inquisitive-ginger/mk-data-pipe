import time
# from MKDirectionDetect import MKDirectionDetect
# import mxnet as mx

class MKEnv(object):
    def __init__(self, capture_instance, mk_serial, num_episodes, learning_steps, display_count, directionDetectModel):
        self.episodes = num_episodes  # Number of episodes to be played
        self.learning_steps = learning_steps  # Maximum number of learning steps within each episodes
        self.display_count = display_count  # The number of episodes to play before showing statistics.
        
        self.capture = capture_instance
        # self.action_space = ['accel', 'brake', 'left_accel', 'right_accel']
        self.action_space = ['accel', 'left_accel', 'right_accel']
        self.mk_serial = mk_serial
        self.reset_commands = ['-', 's', 'j']
        # Loading model to predict direction of Mario
        self.directionDetectModel = directionDetectModel
        # self.directionModel = MKDirectionDetect()
        # self.directionModel.dirNet.load_params("./model.params", ctx=mx.cpu())

    def reset(self):
        for command in self.reset_commands:
            self.mk_serial.send_command(command)
            time.sleep(1)
        print('Resetting environment...')
        time.sleep(10)
        print('Let\'s do this thing!!!')
        bundle = self.capture.set_frame_bundle() # Set a bundle of frames available in our self.capture instance
        # print('bundle of frames: ', bundle) #this verifies that we have a pack of frames in our self.capture instance
        # return self.capture.get_transposed_frames() # this grabs the frame_bundle and transposes it
        return self.capture.get_transposed_frame()
    
    def step(self, action, step_number):
        # backwards_frame used to detect the direction
        bundle = self.capture.set_frame_bundle()
        transposed_frame = self.capture.get_transposed_frame()
        
        before_detect = time.time()
        backwards = False
        direction = self.directionDetectModel.classify_direction(bundle[-1])
        if direction == 'backward':
            backwards = True
        after_detect = time.time()
        # print("DETECT TIME: {}".format(after_detect - before_detect))
        
        # if we're on step 1,2 or 3 we can't possibly be going backwards so we didn't detect backwards
        # this fixes instances where we get negative rewards on early steps
        if step_number > 3 and backwards:
            reward = -100
        else: 
            reward = 1 / self.capture.calc_optical_flow()

        return transposed_frame, reward, self.detect_done()

    def detect_done(self):
        return False