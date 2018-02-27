import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import mxnet as mx
from mxnet import nd

class VideoDataLoader(object):
    def __init__(self, **kwargs):
        super(VideoDataLoader, self).__init__()

        self._files = kwargs.get('files') # input files
        self._pack_size = kwargs.get('pack_size') # number of frames per pack
        self._batch_size = kwargs.get('batch_size')
        self._frame_packs = [] # packs of frames that have been transformed
        self._labels = [] # labels for each pack of framse

        # step though each video to load frames
        for fname, label in self._files:
            self.load_frames(fname, label)

        # convert to ndarrays for use in DataIterator
        self._frame_packs = nd.array(self._frame_packs)
        self._labels = nd.array(self._labels)
        
    # crop and resize frame
    def transform_frame(self, frame):
        start_y = 300
        start_x = 340
        width = 600
        height = 400

        frame = frame.astype(np.float32)/255 # squish pixel values between [0, 1]
        frame = frame[start_y:start_y+height, start_x:start_x+width] # crop image to grab just mario and immediate surroundings
        frame = cv2.resize(frame, (math.ceil(width/4), math.ceil(height/4))) # scale image

        return frame

    # grab all of the frames from a video, transform them
    # and then pack them into groups of 'self._pack_size'
    def load_frames(self, fname, label, frame_count=math.inf):
        print("Loading %s ..." % fname)
        vidcap = cv2.VideoCapture(fname)
        curr_pack = []
        success = True
        index = 0
        pack_index = 0
        while success and index < frame_count:
            success, frame = vidcap.read() # returns frame as numpy ndarray
            if success:
                curr_pack.append(self.transform_frame(frame))
                index += 1
                pack_index += 1
                if pack_index == self._pack_size: # append this pack tp list
                    self._frame_packs.append(curr_pack)
                    self._labels.append(label)
                    curr_pack = []
                    pack_index = 0

    # show subplots of frames
    def plot_frames(self, start_index, num_frames):
        number_of_cols = 2
        number_of_rows = math.ceil(num_frames / 2)
        index = start_index

        fig, axs = plt.subplots(number_of_rows, number_of_cols, figsize=(10,10))
        
        for row in range(number_of_rows):
            for col in range(number_of_cols):
                img = self._video_frames[index]
                axs[row, col].imshow(img.asnumpy())
                axs[row, col].set_title('Frame {}'.format(index))
                index += 1

        plt.show()

    # plot a single frame
    def plot_frame(self, frame):
        plt.imshow(frame.asnumpy())
        plt.show()

    # shuffle data and labels in unison
    def unison_shuffle(self, data, labels):
        assert len(data) == len(labels)
        p = np.random.permutation(len(data))
        return data[p], labels[p]

    # return iterator for training and testing set
    def create_iterators(self, train_percent):
        shuffled_data, shuffled_labels = self.unison_shuffle(self._frame_packs, self._labels)
        data_length = len(shuffled_data)

        # training set
        train_index_end = math.ceil((train_percent / 100) * data_length)
        train_data = shuffled_data[0:train_index_end]
        train_labels = shuffled_labels[0:train_index_end]
        train_data = mx.nd.transpose(train_data, (0,4,1,2,3))
        train_iter = mx.io.NDArrayIter(train_data, train_labels, batch_size=self._batch_size, 
                                       shuffle=True, last_batch_handle='discard') 

        # testing set
        test_data = shuffled_data[train_index_end:]
        test_labels = shuffled_labels[train_index_end:]
        test_data = mx.nd.transpose(test_data, (0,4,1,2,3))
        test_iter = mx.io.NDArrayIter(test_data, test_labels, batch_size=self._batch_size, 
                                       shuffle=True, last_batch_handle='discard') 


        print("Train Iter: ", train_iter.provide_data, train_iter.provide_label)
        print("Test Iter: ", test_iter.provide_data, test_iter.provide_label)
        

        return train_iter, test_iter

    # return a data iterator that releases batches of data
    def get_iterators(self):
        return self.create_iterators(train_percent=70)