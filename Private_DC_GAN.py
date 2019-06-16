#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os, time, itertools, imageio, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy.optimize import bisect


# In[3]:


from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer


# In[4]:


target_eps = sys.argv[1]
trial_num = sys.argv[2]


# In[22]:


# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 20

# G optimizer
num_microbatches=50
l2_norm_clip=1.5


# In[23]:


orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

sampling_probability = batch_size / 55000
steps = train_epoch * 55000 // batch_size

delta = 1e-5
def find_eps(multiplier):
    rdp = compute_rdp(q=sampling_probability,
                  noise_multiplier=multiplier,
                  steps=steps,
                  orders=orders)
    return(get_privacy_spent(orders, rdp, target_delta=delta)[0]-float(target_eps))

noise_multiplier = bisect(find_eps,0.5,3.0)

rdp = compute_rdp(q=sampling_probability,
                  noise_multiplier=noise_multiplier,
                  steps=steps,
                  orders=orders)

epsilon = get_privacy_spent(orders, rdp, target_delta=delta)[0]


# In[3]:


dev = 0.5


# In[5]:


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)

        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)
        noise1 = tf.random_normal(shape=tf.shape(lrelu1),mean=0.0,stddev=dev)
        lrelu1 = tf.add(lrelu1,noise1)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        noise2 = tf.random_normal(shape=tf.shape(lrelu2),mean=0.0,stddev=dev)
        lrelu2 = tf.add(lrelu2,noise2)
        
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        noise3 = tf.random_normal(shape=tf.shape(lrelu3),mean=0.0,stddev=dev)
        lrelu3 = tf.add(lrelu3,noise3)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        noise4 = tf.random_normal(shape=tf.shape(lrelu4),mean=0.0,stddev=dev)
        lrelu4 = tf.add(lrelu4,noise4)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)

        return o, conv5


# In[7]:


fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# In[9]:


# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])


# In[10]:


# variables : input
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real+D_loss_fake

vector_G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1]))
G_loss = tf.reduce_mean(vector_G_loss)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]


# In[11]:


ledger = privacy_ledger.PrivacyLedger(
          population_size=55000,
          selection_probability=(batch_size/55000),
          max_samples=1e6,
          max_queries=1e6)

G_optimizer = dp_optimizer.DPAdamGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=lr,
        beta1=0.5,
        ledger=ledger)


# In[12]:


# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim= G_optimizer.minimize(loss=vector_G_loss, var_list=G_vars)


# In[13]:


# open session and initialize all variables
saver = tf.train.Saver()
sess = tf.InteractiveSession()


# In[ ]:


tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'Eps_'+str(target_eps)+'_Trial'+str(trial_num)+'_Private_MNIST_DCGAN_results/'
model = 'Private_MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(mnist.train.num_examples // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([vector_G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)


# In[ ]:


save_path = saver.save(sess, root+"model.ckpt")
print("Model saved in path: %s" % save_path)


# In[ ]:


from sklearn.decomposition import PCA
import scipy as sp

train_set = np.resize(train_set,[55000,64*64])

true_pca = PCA(n_components=20)
true_pca.fit(train_set)
true_subspace = true_pca.components_
true_component_1 = true_subspace[0,:]
true_component_2 = true_subspace[1,:]
true_component_3 = true_subspace[2,:]

def generate_sample(size):
    z_ = np.random.normal(0,1,(size,1,1,100))
    test_images = sess.run(G_z, {z: z_, isTrain: False})
    return np.resize(test_images,[size,64*64])

def calc_mean(array):
    mean = np.mean(array)
    sd = np.std(array)
    interval = np.zeros(2)
    interval[0] = mean-1.96*(sd/np.sqrt(len(array)))
    interval[1] = mean+1.96*(sd/np.sqrt(len(array)))
    return(mean,interval)

def evaluate_performance(trials):
    
    prim_dist = np.zeros(trials)
    second_dist = np.zeros(trials)
    third_dist = np.zeros(trials)
    geo_dist = np.zeros(trials)
    
    for i in range(trials):
        
        # Draw a fresh sample from the model and perform PCA
        sample = np.zeros([55000,64*64])
        
        for j in range(550):
            sample[0+100*j:100+100*j,:] = generate_sample(100)

        synthetic_pca = PCA(n_components=20)
        synthetic_pca.fit(sample)
        synthetic_subspace = synthetic_pca.components_
        
        # Calc distance between the first 3 principal components
        prim_dist[i] = np.linalg.norm(true_component_1-synthetic_subspace[0,:])
        second_dist[i] = np.linalg.norm(true_component_2-synthetic_subspace[1,:])
        third_dist[i] = np.linalg.norm(true_component_3-synthetic_subspace[2,:])

        # Calc Geodesic distance on Grasmanian
        angles = sp.linalg.subspace_angles(true_subspace,synthetic_subspace)
        geo_dist[i] = np.linalg.norm(angles)
        
    prim_mean, prim_interval = calc_mean(array=prim_dist)
    second_mean, second_interval = calc_mean(array=second_dist)
    third_mean, third_interval = calc_mean(array=third_dist)
    geo_mean, geo_interval = calc_mean(array=geo_dist)
    
    return(prim_mean,prim_interval,second_mean,second_interval,
           third_mean,third_interval,geo_mean,geo_interval)

print('Evaluation Start!')
start_time = time.time()
a, b, c, d, e, f, g, h = evaluate_performance(10)
end_time = time.time()
elapsed_time = end_time - start_time
print('ptime: %.2f' % (elapsed_time))
print('The mean distance between the first principal component of the two subspaces is: ' + str(a))
print('The confidence interval of this mean is: ' + str(b))
print('The mean distance between the second principal component of the two subspaces is: ' + str(c))
print('The confidence interval of this mean is: ' + str(d))
print('The mean distance between the third principal component of the two subspaces is: ' + str(e))
print('The confidence interval of this mean is: ' + str(f))
print('The mean geodesic distance between the two subspaces is: ' + str(g))
print('The confidence interval of this mean is: ' + str(h))


# In[24]:


print('epsilon = ' + str(epsilon) + '\ndelta = ' + str(delta) + 
      '\nnoise multiplier = ' + str(noise_multiplier))


# In[ ]:


sess.close()

