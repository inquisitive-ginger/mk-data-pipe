import mxnet as mx
from mxnet import nd
from mxnet import gluon

class Net(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(16, kernel_size=5, strides=2)
            self.bn1 = gluon.nn.BatchNorm()
            self.conv2 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn2 = gluon.nn.BatchNorm()
            self.conv3 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn3 = gluon.nn.BatchNorm()
            #self.lstm = gluon.rnn.LSTMCell(128)
            self.dense1 = gluon.nn.Dense(128, activation='relu')
            self.dense2 = gluon.nn.Dense(64, activation='relu')
            self.action_pred = gluon.nn.Dense(available_actions_count)
            self.value_pred = gluon.nn.Dense(1)
            #self.states = self.lstm.begin_state(batch_size=1, ctx=ctx)

    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = nd.flatten(x).expand_dims(0)
        #x, self.states = self.lstm(x, self.states)
        x = self.dense1(x)
        x = self.dense2(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values