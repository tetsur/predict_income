
#!/usr/bin/env python

import argparse
import os
from chainer.datasets import mnist
import chainer
from chainer import training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.backends import cuda
from chainer import serializers
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import net


def main():
    parser = argparse.ArgumentParser(description='Chainer: NN')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='path/to/output',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=200, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    model = net.NN(5,50,30,2)
    if 0 <= args.gpu:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # GPUを使うための処理
    model = L.Classifier(model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # モデルの読み込み npzはnumpy用
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)
    
    workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked","?"]
    education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th","Preschool","?"]
    occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                  "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"]


    #地価ランキングのリスト化
    data = pd.read_table("train.tsv", sep="\t", usecols=["age", "workclass", "education", "occupation", "sex", "Y"])
    data = data.replace({'Female':0, 'Male': 1})
    for i, v in enumerate(workclass):
        data = data.replace(v, i)
    for i, v in enumerate(education):
        data = data.replace(v, i)
    for i, v in enumerate(occupation):
        data = data.replace(v, i)
    print(data)
    train = data.select_dtypes(include=int).values
    print(train)
    data = data.replace({'>50K': 0, '<=50K': 1})
    lab = data.iloc[:, 5]
    print(lab)
    lab = np.array(lab.astype('int32'))
    train = np.array(train.astype('float32'))
   
    dataset = list(zip(train,lab))

    train, test = train_test_split(dataset, test_size=0.2)
  

#------------------イテレーターによるデータセットの設定-----------------------------------
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
#---------------------------------------------------------------

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='my_log_data'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/accuracy', 'validation/main/accuracy','main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='loss.png'))
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # トレーナーの実行
    trainer.run()

    serializers.save_npz("NN.npz", model)



if __name__ == '__main__':
    main()
    
