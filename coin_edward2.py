import math
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow_probability import edward2 as ed

n_steps = 2000

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(1)
for _ in range(4):
    data.append(0)

def model(num_data):
    # define the hyperparameters that control the beta prior
    alpha0 = 10.0
    beta0 = 10.0
    # sample f from the beta prior
    f = ed.Beta(alpha0, beta0, name="latent_fairness")
    # sample the data
    data = ed.Bernoulli(probs=f, name="obs", sample_shape=num_data)
    return data

log_joint = ed.make_log_joint_fn(model)

# create tf.Variables for the variational parameters
alpha_q = tf.exp(tf.Variable(tf.log(15.0), name="log_alpha_q"))
beta_q = tf.exp(tf.Variable(tf.log(15.0), name="log_beta_q"))

def guide(alpha_q, beta_q):
    # sample latent fairness from the distribution
    f = ed.Beta(alpha_q, beta_q, name="latent_fairness_q")
    return f

log_q = ed.make_log_joint_fn(guide)

# set up the variational objective
f = guide(alpha_q, beta_q)
energy = log_joint(latent_fairness=f, obs=data, num_data=len(data))
entropy = -log_q(latent_fairness_q=f, alpha_q=alpha_q, beta_q=beta_q)
elbo = energy + entropy

# set up the optimizer
optimizer = AdamOptimizer(learning_rate = 0.0005, beta1 = 0.90, beta2 = 0.999)

# set up the training op
train = optimizer.minimize(-elbo)

# do gradient steps
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(n_steps):
        sess.run(train)
        if step % 100 == 0:
            print(sess.run(-elbo))

    alpha_q = sess.run(alpha_q)
    beta_q = sess.run(beta_q)

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
