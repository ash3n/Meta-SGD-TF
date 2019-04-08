import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import utils as ff


print('loading data..')
omniglot = ff.sqds.load('data/omniglot.sqds')
omni_arr = np.array(omniglot)
print(omni_arr.shape)


# static alphas ( original setup )

#width = 20
#n_classes = 5

batch_size = 4
bs = batch_size

n_inputs = 400
n_hnodes = 200
n_outputs = 5

# initializes parameters of a layer
def pvar(shape, name):
    _thetas = ff.tvar(shape, 'thetas_'+name)
    _alphas = ff.tvar(shape, 'alphas_'+name)
    #_alphas.assign_add(0.1)
    thetas.append(_thetas)
    alphas.append(_alphas)
    return [_thetas, tf.zeros(shape), _alphas]

thetas = []
alphas = []
V_d1 = pvar([n_inputs+1,n_hnodes], 'd1')
V_d2 = pvar([n_hnodes+1,n_outputs], 'd2')
layers = [V_d1, V_d2]
for layer in layers:
    layer[1] = tf.zeros([batch_size]+list(layer[1].shape))
    layer[2].assign(tf.ones(layer[2].shape))

# single forward pass through network
def forward(x): # input shape (b, d)
    h_d0 = tf.expand_dims(x, 1)
    h_d1 = ff.dense(h_d0, V_d1[0]+V_d1[1]*V_d1[2], activation=tf.nn.leaky_relu) #*V_d1[2]
    h_d2 = ff.dense(h_d1, V_d2[0]+V_d2[1]*V_d2[2]) #*V_d2[2]
    return tf.reduce_sum(h_d2, 1)

# calculate prediction error
def inner_loss(h, y): # input shape (b, d)
    #print(tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(h, -1), tf.argmax(y, -1)))))
    #return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(y-h), -1)))
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=h)

optimizer = tf.train.AdamOptimizer(1e-2)
def train_step(X, Y): # input shape (l, b, d)
    with tf.GradientTape(persistent=True) as tape:
        seq_len = int(X.shape[0])
        _batch_size = int(X.shape[1])
        L = []
        
        o = tf.zeros([0,_batch_size,n_outputs])
        for i in range(seq_len):
            h = forward(X[i])
            o = tf.concat([o, tf.expand_dims(h, 0)], 0)
            J = inner_loss(h, Y[i])
            global inner_grads
            inner_grads = tape.gradient(J, thetas)
            for layer, grad in zip(layers, inner_grads):
                #layer[1] -= tf.stop_gradient(grad) * 1e-2 * tf.sigmoid(layer[2])
                #layer[1] -= tf.stop_gradient(grad) * 1e-2
                layer[1] = tf.stop_gradient(layer[1] - grad*1e-2)
            L.append(J)
        outer_loss = tf.reduce_mean(L)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(o, -1), tf.argmax(Y, -1))))
        
    outer_grads = tape.gradient(outer_loss, thetas+alphas)
    optim_grads = [tf.minimum(tf.maximum(grad, -40), 40) for grad in outer_grads]
    #for theta, grad in zip(thetas+alphas, outer_grads):
        #theta.assign_add(-1 * grad * 1e-2)
    optimizer.apply_gradients(zip(optim_grads, thetas+alphas))

    for layer in layers:
        #layer[0].assign_add(layer[1])
        layer[1] *= 0
    
    return outer_loss, acc

print('network compiled')


omni_batches = tf.transpose(np.reshape(omni_arr[:len(omni_arr)-len(omni_arr)%bs], [-1,bs,100,405]), [0, 2, 1, 3])

log_dir = 'mg/tb'
tb_tag = '_msgd-1'

global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(log_dir)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for epoch in range(400):
        for i, v in enumerate(omni_batches):
            X1 = tf.to_float(v[:,:,5:])
            y1 = tf.to_float(v[:,:,:5])

            print(epoch+1, '(%d/%d) '%(i+1, omni_batches.shape[0]), end='')
            loss_train, acc_train = train_step(X1, y1)

            tf.contrib.summary.scalar('accuracy_train'+tb_tag, acc_train)
            tf.contrib.summary.scalar('loss_train'+tb_tag, loss_train)
            global_step.assign_add(bs)

            print("acc %.4f loss %.4f"%(acc_train, loss_train))
