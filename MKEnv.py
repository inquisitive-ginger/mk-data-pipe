import time
# from MKVideoCapture import MKVideoCapture

class MKEnv(object):
    def __init__(self, capture_instance, mk_serial, num_episodes, learning_steps, display_count):
        self.episodes = num_episodes  # Number of episodes to be played
        self.learning_steps = learning_steps  # Maximum number of learning steps within each episodes
        self.display_count = display_count  # The number of episodes to play before showing statistics.
        
        self.capture = capture_instance
        self.action_space = ['accel', 'brake', 'left_accel', 'right_accel']
        self.mk_serial = mk_serial
        self.reset_commands = ['-', 's', 'j']

    def reset(self):
        for command in self.reset_commands:
            self.mk_serial.send_command(command)
            time.sleep(1)
        print('Sleeping for 10...')
        time.sleep(10)
        print('Awake now!')
        self.capture.set_frame_bundle()
        return self.capture.get_transposed_frames()
    
    def step(self, action):
        self.capture.set_frame_bundle()
        transposed_frames = self.capture.get_transposed_frames()
        optical_flow = self.capture.calc_optical_flow()
        return transposed_frames, optical_flow, False