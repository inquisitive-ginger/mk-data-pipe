import math
import cv2
import numpy as np
import mxnet as mx

class MKVideoCapture:
    def __init__(self, camera_number, bundle_size):
        self.capture = cv2.VideoCapture(camera_number)
        self.bundle_size = bundle_size
        self.frame_width = 1920
        self.frame_height = 650
        self.frame_bundle = []
        self.resize_ratio = 4

    def transform_frame(self, frame):
        start_y = 480
        start_x = 0

        # frame = frame.astype(np.float32)/255 # squish pixel values between [0, 1]
        frame = frame[start_y:start_y+self.frame_height, start_x:start_x+self.frame_width] # crop image to grab just mario and immediate surroundings
        frame = cv2.resize(frame, (self.frame_width//self.resize_ratio, self.frame_height//self.resize_ratio)) # scale image
        
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        
        return frame
    
    def set_frame_bundle(self):
        frames = []
        index = 0
        while index < self.bundle_size:
            ret, frame = self.capture.read()
            if ret:
                # Next line is needed to detect direction
                if index == self.bundle_size - 1: last_frame = frame
                frame = self.transform_frame(frame)
                frames.append(frame)
            index += 1
            
        self.frame_bundle = frames
        return last_frame

    def get_transposed_frames(self):
        # mxArray = mx.nd.empty((self.bundle_size, self.frame_height, self.frame_width, 3))
        mxArray = mx.nd.array(self.frame_bundle)
        mxArray = mxArray[0:self.bundle_size, :, :, :].transpose((0, 3, 2, 1)).reshape((-1, self.frame_width//self.resize_ratio, self.frame_height//self.resize_ratio)) # reshaping
        mxArray = mxArray.expand_dims(0) # expanding to (1, bundle_size*3, height, width)
        return mxArray

    def calc_optical_flow(self):
        frame1 = self.frame_bundle[0]
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # prvs = frame1
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        optical_flow_sum = 0
        for frame in self.frame_bundle[1:]:
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # next = frame

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            optical_flow_sum += np.mean(mag)
            prvs = next

        return optical_flow_sum