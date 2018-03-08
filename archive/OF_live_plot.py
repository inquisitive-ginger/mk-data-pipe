from time import time
import threading
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from drawnow import drawnow

from optical_flow import calc_optical_flow, get_frames, transform_frame

plt.ion() # enable interactivity
fig = plt.figure()  # make a figure
start_time = time()
timepoints = []
data = []
flows = []
yrange = [0, 20000]
window = 5 # seconds of data to view at once

frames = []
optical_flows = []
num_frames = 5
def acquire_video():
    # connect to capture card stream
    capture = cv2.VideoCapture(1)

    # grab frames and put them into a buffer as they come in
    while True:
        ret, frame = capture.read(1)
        if ret:
            frame = transform_frame(frame)
            frames.append(frame)
            if len(frames) % num_frames == 0:
                plot_optical_flow(frames[-1*num_frames:])
                pass

            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    return frames

def plot_optical_flow(frames):
    flow = calc_optical_flow(frames)
    data.append(flow)
    timepoints.append(time() - start_time)
    drawnow(make_fig)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# callback to drawnow() for rendering updated plot 
def make_fig():
    plt.xlabel('Time (s)', fontsize='14', fontstyle='italic')
    plt.ylabel('Optical Flow', fontsize='14', fontstyle='italic')
    plt.axes().grid(True)
    plt.plot(timepoints, data, linestyle='-')

    current_time = timepoints[-1]
    if current_time > window:
        plt.xlim([current_time-window, current_time])
    else:
        plt.xlim([0, window])
    
def main():
    acquire_video()

if __name__ == '__main__':
    main()

