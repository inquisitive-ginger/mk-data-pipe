import math
import cv2
import numpy as np

def transform_frame(frame):
    start_y = 720
    start_x = 0
    width = 1920
    height = 310

    # frame = frame.astype(np.float32)/255 # squish pixel values between [0, 1]
    frame = frame[start_y:start_y+height, start_x:start_x+width] # crop image to grab just mario and immediate surroundings
    # frame = cv2.resize(frame, (math.ceil(width/4), math.ceil(height/4))) # scale image

    return frame

def get_frames(capture):
    frames = []
    ret, frame = capture.read()
    frames.append(transform_frame(frame))
    index = 0
    while index < capture.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = capture.read()
        if(ret):
            frame = transform_frame(frame)
            print(capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(frame)
            print(frame.shape)
        index += 1

    capture.release()
    return frames

def calc_optical_flow(frames):
    frame1 = frames[0]
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    optical_flow_sum = 0
    for frame in frames[1:]:
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        optical_flow_sum += mag.sum()
        prvs = next

    return optical_flow_sum

def main():
    labels = ['IDLE', 'SLOW', 'FAST']
    files = ['./data/idle_clipped.mp4', './data/gd_slow_lap01.mp4', './data/gd_fast_lap01.mp4']
    frames = {}

    for i, file in enumerate(files):
        print("Loading %s ..." % file)
        cap = cv2.VideoCapture(file)
        frames[labels[i]] = get_frames(cap)

    total_completely_correct = 0
    total_partially_correct = 0
    num_frames = 10
    sample_size = 1000
    for i in range(sample_size):
        flows = np.empty(len(files))
        for j, label in enumerate(labels):
            rnd_start_frame = np.random.randint(0, len(frames[label]) - num_frames)
            of_frames = frames[label][rnd_start_frame:rnd_start_frame+num_frames]
            optical_flow_sum = calc_optical_flow(of_frames)
            flows[j] = optical_flow_sum
            print("%s : %d" % (label, optical_flow_sum))

        completely_correct = all(flows[i] < flows[i+1] for i in range(len(flows)-1))
        partially_correct = flows[0] < flows[len(flows)-1]
        if completely_correct:
            total_completely_correct += 1
        
        if partially_correct:
            total_partially_correct += 1

    percent_totally_correct = 100 * (total_completely_correct / float(sample_size))
    percent_partially_correct = 100 * (total_partially_correct / float(sample_size))
    
    print("Percent Totally Correct: %d" % percent_totally_correct)
    print("Percent Partially Correct: %d" % percent_partially_correct)

if __name__ == '__main__':
    main()