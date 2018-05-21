import tensorflow as tf

ALPHASIZE = 98
CELLSIZE = 512
NLAYERS = 3
SEQLEN = 30

Xd = tf.placeholder(tf.uint8, [None, None])
X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)
Yd = tf.placeholder(tf.uint8, [None, None])
Y_ = tf.one_hot(Yd, ALPHASIZE, 1.0, 0.0)
Hin = tf.placeholder(tf.float32, [None, CELLSIZE*NLAYERS])

cell = tf.nn.rnn_cell.GRUCell(CELLSIZE)
mcell = tf.nn.rnn_cell.MultiRNNCell([cell]*NLAYERS, state_is_tuple=False)
Hr, H = tf.nn.dynamic_rnn(mcell,X, initial_state=Hin)

Hf = tf.reshape(Hr, [-1, CELLSIZE])
Ylogits = tf.contrib.layers.linear(Hf, ALPHASIZE)
Y = tf.nn.softmax(Ylogits)
Yp = tf.arg_max(Y, 1)
Yp = tf.reshape(Yp, [BATCHSIZE, -1])

loss = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

for epoch in range(20):
    inH = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
    for x, y_ in utils.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=30):
        dic = {X:x, Y_:y_, Hin:inH}
        _,y,outH = sess.run([train_step,Yp,H], feed_dict=dic)
        inH = outH


with tf.Session as sess:
    resto = tf.train.import_meta_graph('shake_200.meta')
    resto.restore(sess, 'shake_200')

    x = np.array([[0]])
    h = np.zeros([1, INTERNALSIZE*NLAYERS], dtype=tf.float32)

    for i in range(10000):
        dic = {'X:0':x, 'Hin:0':h, 'batchsize:0':1}
        y, h = sess.run(['Y:0', 'H:0'], feed_dict=dic)
        c = my_txtutils.sample_from_probabilities(y, topn=5)
        x = np.array([[c]])

        print(chr(my_txtutils.convert_to_ascii(c)), end="")
