import numpy as np
import cPickle as pickle
import tensorflow as tf
import time
#%matplotlib inline
import matplotlib.pyplot as plt
import math

def load_data():
    f = open('../data/XY_oneclick_np_100','rb')
    #x = np.asarray(x)
    #y = np.asarray(y)

    #features = []

    x=pickle.load( f)
    y=pickle.load( f)
    features = pickle.load(f)

    new_x = [[0 if fea is None else fea  for fea in case] for case in x]
    x = np.asarray(new_x)

    indices = range(len(y))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]


    '''fout = open('../data/XY.txt', 'w')
    for i in x:
        for j in i:
            print>>fout, j,'\t',
        print>>fout, ''

    fout.close()'''




    train_size = int(len(x)*0.7)
    dev_size = int(len(x) * 0.3)

    train_x = x[:train_size]
    train_y = y[:train_size]
    dev_x = x[train_size:train_size + dev_size]
    dev_y = y[train_size:train_size + dev_size]
    test_x = x[train_size + dev_size:]
    test_y = y[train_size + dev_size:]
    return train_x, train_y, dev_x, dev_y, test_x, test_y, features

def predict(sess,childinput_x, childinput_y, childaccuracy, datax, datay, batchnum):
    childvalid_acc = 0.0

    for i in range(batchnum):
        child_np_accuracy = sess.run(childaccuracy, feed_dict={
            childinput_x: datax[i * childbatch_size: (i + 1) * childbatch_size],
            childinput_y: datay[i * childbatch_size: (i + 1) * childbatch_size]
        })
        childvalid_acc += child_np_accuracy
        #print child_np_accuracy

    childvalid_acc = 1.0 * childvalid_acc / len(datay)
    return childvalid_acc

train_x, train_y, dev_x, dev_y, test_x, test_y, features = load_data()

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


Maximum_layer_num = 5
childbatch_size = 32
childtrainbatchnum = int(np.ceil(1.0*len(train_x)/childbatch_size))
childvalidbatchnum = int(np.ceil(1.0*len(dev_x)/childbatch_size))

with tf.Session(graph=tf.Graph()) as childsess:
    with tf.variable_scope("childDNN_", reuse=False):
        childinput_x = tf.placeholder(tf.float32, [None, len(features)], name="childinput_x")
        childinput_y = tf.placeholder(tf.int32, [None], name="childinput_y")
        childnsamples = childinput_y.shape[0]
        child_previous_layersize = len(features)
        childlayer = childinput_x

        # print tf.get_variable_scope()
        layer_sizes = []
        # child_params = []
        np_chosen_classes = [10,10]
        for time_step, chosen_class in enumerate(np_chosen_classes):
            layer_size = len(features) * (chosen_class + 1) / 10
            layer_sizes.append(layer_size)
            childW = tf.Variable(tf.truncated_normal([child_previous_layersize, layer_size],
                                                     stddev=1.0 / math.sqrt(float(child_previous_layersize))),
                                 name="W" + str(time_step))
            childb = tf.Variable(tf.zeros([layer_size]), name="b" + str(time_step))

            variable_summaries(childW)
            variable_summaries(childb)

            # child_params.append(childW)
            # child_params.append(childmul)
            childlayer = tf.tanh(tf.matmul(childlayer, childW)+ childb)
            #childlayer = tf.layers.dropout(childlayer)

            child_previous_layersize = layer_size

        childoutW = tf.Variable(tf.truncated_normal([child_previous_layersize, 2],
                                                    stddev=1.0 / math.sqrt(float(child_previous_layersize))),
                                name="Wout")
        childoutb = tf.Variable(tf.zeros([2]), name="outb")

        variable_summaries(childoutW)
        variable_summaries(childoutb)

        childscore = tf.matmul(childlayer, childoutW) + childoutb
        #childprobability = tf.nn.softmax(childscore)

        # print(childscore, childinput_y)
        print layer_sizes

        childloglik = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=childscore, labels=childinput_y)
        # childloglik = tf.log(childprobability[tf.range(childbatch_size), childinput_y[:,0]])



        childloss = tf.reduce_mean(childloglik)
        tf.summary.scalar('loss', childloss)

        childadam = tf.train.AdamOptimizer(learning_rate=0.001)  # Our optimizer
        childtrainop = childadam.minimize(childloss)

        childlabel = tf.cast(tf.argmax(childscore, axis=1), tf.int32)

        childaccuracy = tf.reduce_sum(tf.cast(tf.equal(childlabel, childinput_y), tf.float32))

        tf.summary.scalar('accuracy', childaccuracy)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', childsess.graph)
        childsess.run(tf.global_variables_initializer())

        # child_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='childRNN')
        # print child_params
        # sess.run(tf.initialize_variables(child_params))
        '''print childsess.run(
            [childlabel,childinput_y,tf.equal(childlabel, childinput_y),childaccuracy], feed_dict={
                childinput_x: train_x[0: childbatch_size],
                childinput_y: train_y[0: childbatch_size]
            })'''



        child_acc_epoch = []
        for childepoch in range(100):
            batchloss = []
            for i in range(childtrainbatchnum):
                summary, _, child_np_loss_value = childsess.run([merged, childtrainop, childloss], feed_dict={
                    childinput_x: train_x[i * childbatch_size: (i + 1) * childbatch_size],
                    childinput_y: train_y[i * childbatch_size: (i + 1) * childbatch_size]
                })
                #print 'batch ', i, 'loss:', child_np_loss_value
                batchloss.append(child_np_loss_value)
                train_writer.add_summary(summary, i)
                #  print 'Cost: ', child_np_loss_value
            childtrain_acc = predict(childsess, childinput_x, childinput_y, childaccuracy, train_x, train_y,
                                     childtrainbatchnum)
            childvalid_acc = predict(childsess, childinput_x, childinput_y, childaccuracy, dev_x, dev_y,
                                     childvalidbatchnum)

            print 'Child Epoch', childepoch,'loss=',sum(batchloss)/len(batchloss), ': train ', childtrain_acc, 'valid ', childvalid_acc

            child_acc_epoch.append(childvalid_acc)

        reward = sum(child_acc_epoch) / len(child_acc_epoch)