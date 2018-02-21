from MKVideoCapture import MKVideoCapture

class MKEnv(object):
    def __init__(self, bundle_size):
        self.capture = MKVideoCapture(1, bundle_size)
        self.action_space = ['left', 'right', 'left_accel', 'right_accel', 'accel', 'none']

    def reset(self):
        self.capture.set_frame_bundle()
        return self.capture.get_transposed_frames()
    
    def step(self, action):
        self.capture.set_frame_bundle()
        transposed_frames = self.capture.get_transposed_frames()
        optical_flow = self.capture.calc_optical_flow()
        return transposed_frames, optical_flow, False
