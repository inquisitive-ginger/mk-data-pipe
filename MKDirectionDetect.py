#import Dependencies
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import time
import os
import logging
logging.basicConfig(level=logging.INFO)
from mxnet.gluon.model_zoo import vision as models
from mxnet.image import color_normalize
import cv2
from plotImages import PlotImages

batch_size = 64
ctx = [mx.cpu()]

class MKDirectionDetect(object):
    def __init__(self, ctx=ctx):
        # lets use a pretrained squeezenet, this a model known for being decently good accuracy at a low computational cost
        squeezenet = models.squeezenet1_1(pretrained=True, prefix="direction_", ctx=ctx)

        # create a new copy of squeezenet, this time though only have 2 output classes (hotdog or not hotdog)
        self.dirNet = models.squeezenet1_1(classes=2, prefix="direction_")
        self.dirNet.collect_params().initialize(ctx=ctx)

        # use the the features chunk of squeezenet, only leave the output untouched
        self.dirNet.features = squeezenet.features

        # in the trainer, specify that we only want to update the output chunk of self.dirNet
        self.trainer = gluon.Trainer(self.dirNet.output.collect_params(), 'sgd', {'learning_rate': .01})
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

        self.metric = mx.metric.create(['acc', 'f1'])

    # given guess z and label y, compute the loss
    def unbalanced_loss(self, loss_func, z, y):
        # there are 3 times more images of forward than going backward :(
        positive_class_weight = 3
        regular_loss = loss_func(z, y)
        # convienently y is either 1 (wrong Direction) or 0 (right direction) so scaling is pretty simple
        scaled_loss = regular_loss * (1 + y.astype("float32")*positive_class_weight)/positive_class_weight
        return scaled_loss

    # return metrics string representation
    def metric_str(self, names, accs):
        return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

    def transform(self, data, label):
        data = data.astype("float32")
        data = data/255
        data = data.transpose([2,1,0])
        data = color_normalize(data,
                                mean=mx.nd.array([0.485, 0.456, 0.406]).reshape([3,1,1]),
                                std=mx.nd.array([0.229, 0.224, 0.225]).reshape([3,1,1]))
        return data, label
    
    def evaluate(self, net, data_iter, ctx):
        for i, (data,label) in enumerate(data_iter):
            print('batch%d' %i)
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            # print(label)
            outputs = []
            for i, x in enumerate(data):
                print('x number %d' % i)
                outputs.append(net(x))
            self.metric.update(label, outputs)
        out = self.metric.get()
        self.metric.reset()
        return out

    def train(self):
        # Training and validation iterators
        # train_iter = mx.io.ImageRecordIter(batch_size=batch_size, data_shape=(3,270,270), label_width=1, path_imgrec='mk_train.rec', shuffle=True)
        # val_iter = mx.io.ImageRecordIter(batch_size=batch_size, data_shape=(3,270,270), label_width=1, path_imgrec='mk_val.rec', shuffle=False)

        train_dataset = mx.gluon.data.vision.ImageFolderDataset("./videoFrames/", transform=self.transform)
        train_iter = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        epochs = 10
        best_f1 = 0
        log_interval = 100

        for epoch in range(epochs):
            print('epoch #', epoch)
            tic = time.time()
            #train_iter.reset()
            btic = time.time()
            for i, (data, label) in enumerate(train_iter):
                print('batch #:', i)

                # the model zoo models expect normalized images
                data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                with autograd.record():
                    for x, y in zip(data, label):
                        z = self.dirNet(x)
                        # rescale the loss based on class to counter the imbalance problem   
                        L = self.unbalanced_loss(self.loss, z, y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)
                    for L in Ls:
                        L.backward()
                self.trainer.step(data[0].shape[0])
                self.metric.update(label, outputs)
                if log_interval and not (i+1)%log_interval:
                    names, accs = self.metric.get()
                    print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                                epoch, i, batch_size/(time.time()-btic), self.metric_str(names, accs)))
                btic = time.time()

            names, accs = self.metric.get()
            self.metric.reset()
            print('[Epoch %d] training: %s'%(epoch, self.metric_str(names, accs)))
            print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
            val_names, val_accs = self.evaluate(self.dirNet, train_iter, ctx)
            print('[Epoch %d] validation: %s'%(epoch, self.metric_str(val_names, val_accs)))

            if val_accs[1] > best_f1:
                best_f1 = val_accs[1]
                print('Best validation f1 found. Checkpointing...')
                self.dirNet.save_params('direction-%d.params'%(epoch)
    
    def cropAndResizeToSquareImage(self, numpyImage):
        if numpyImage is not None:
            (h,w,c) = numpyImage.shape
            if w > h:
                toRemove = w - h
                croppedImage = numpyImage[:, toRemove//2:w-toRemove//2]

            (h,w,c) = croppedImage.shape
            resizedImage = cv2.resize(croppedImage, (w//4, h//4)) # scale image
            return resizedImage
        else:
            pass

    def classify_direction(self, url):
        I = cv2.imread(url)
        I = nd.array(I)
        image, label = self.transform(I, nd.array([0]))
        image = image.expand_dims(0)
        out = mx.nd.softmax(self.dirNet(image))
        print('Probabilities are: '+str(out[0].asnumpy()))
        result = np.argmax(out.asnumpy())
        outstring = ['forward', 'backward']
        print(outstring[result])
        return outstring

# directionModel = MKDirectionDetect()
# directionModel.dirNet.save_params("./model.params")
# directionModel = MKDirectionDetect(ctx=mx.cpu())
# directionModel.dirNet.load_params("./model.params", ctx=mx.cpu())

# directionModel.classify_direction("./videoFrames/video0/frame254.jpg")
# directionModel.classify_direction("./videoFrames/video1/frame222.jpg")
# directionModel.classify_direction("./videoFrames/video0/frame254.jpg")
# directionModel.classify_direction("./videoFrames/video1/frame824.jpg")
# directionModel.classify_direction("./videoFrames/video0/frame46.jpg")
# directionModel.classify_direction("./videoFrames/video1/frame50.jpg")
# directionModel.classify_direction("./videoFrames/video0/frame234.jpg")
# directionModel.classify_direction("./videoFrames/video1/frame854.jpg")