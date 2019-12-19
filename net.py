
#!/usr/bin/env python
import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class NN(chainer.Chain):

    def __init__(self, n_in, n_h, n_h_2, n_h_3):
        super(NN, self).__init__()
        with self.init_scope():
            # encoder 入力から隠れベクトルの作成
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h_2)
            #隠れベクトルから平均ベクトルの作成
            self.le3 = L.Linear(n_h_2,n_h_3)  # 第１は入力信号数　

    def __call__(self, x):
        #h1 = F.dropout(F.tanh(self.le1(x)), ratio=0.9)
        h1 = F.dropout(F.relu(self.le1(x)),ratio=0.9)
        h2 = F.dropout(F.relu(self.le2(h1)),ratio=0.5)
        h3 = self.le3(h2)
        return h3


