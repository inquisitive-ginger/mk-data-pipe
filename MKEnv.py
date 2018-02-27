import time
from MKVideoCapture import MKVideoCapture

class MKEnv(object):
    def __init__(self, bundle_size, mk_serial):
        self.capture = MKVideoCapture(0, bundle_size)
        self.action_space = ['accel', 'brake', 'sl_accel', 'sr_accel', 'hl_accel', 'hr_accel']
        self.action_map = {0: 1, 1: 0, 2: 5, 3: 3, 4: 13, 5: 11}
        self.mk_serial = mk_serial
        self.reset_commands = ['-', 's', 'j']

    def reset(self):
        for command in self.reset_commands:
            self.mk_serial.send_command(command)
            time.sleep(1)
        print('sleeping for 10')
        time.sleep(10)
        print('awake now')
        self.capture.set_frame_bundle()
        return self.capture.get_transposed_frames()
    
    def step(self, action):
        self.capture.set_frame_bundle()
        transposed_frames = self.capture.get_transposed_frames()
        optical_flow = self.capture.calc_optical_flow()
        return transposed_frames, optical_flow, False