import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
env = gym.make("MountainCar-v0")
env.reset()


class dqn(object):
    def __init__(self):
        self.batch_size = 64
        self.episodes = 20000
        self.input_size = env.observation_space.sample().size
        self.output_size = env.action_space.n
        self.gamma = 0.9
        self.epsilon = 1.0
        self.step = 0
        self.learning_rate = 0.0001
        self.dropout = 1.0
        self.lambda1 = 0.01
        self.initial_epsilon = self.epsilon
        self.final_epsilon = 0.01
        self.weights = {}
        self.biases = {}
        self.target_weights = {}
        self.target_biases = {}
        self.create_nn()
        self.create_training_network()
        self.memory = deque()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def create_nn(self):

        s1 = {1: [self.input_size, 30], 2: [30, 30], 3: [30, self.output_size]}
        s2 = {1: [30], 2: [30], 3: [self.output_size]}
        for i in s1:
            self.weights[i] = tf.Variable(tf.truncated_normal(s1[i]), name='w{0}'.format(i))
            self.biases[i] = tf.Variable(tf.truncated_normal(s2[i]), name='b{0}'.format(i))
            self.target_weights[i] = tf.Variable(tf.truncated_normal(s1[i]), name='tw{0}'.format(i))
            self.target_biases[i] = tf.Variable(tf.truncated_normal(s2[i]), name='tb{0}'.format(i))

    def feed_forward(self, z):
        q = tf.nn.tanh(tf.matmul(z, self.weights[1]) + self.biases[1])
        for i in range(2, len(self.weights), 1):
            q = tf.nn.tanh(tf.matmul(q, self.weights[i]) + self.biases[i])
        q = tf.matmul(q, self.weights[len(self.weights)]) + self.biases[len(self.biases)]
        return q

    def feed_forward_target(self, z):
        q = tf.nn.tanh(tf.matmul(z, self.target_weights[1]) + self.target_biases[1])
        for i in range(2, len(self.weights), 1):
            q = tf.nn.tanh(tf.matmul(q, self.target_weights[i]) + self.target_biases[i])
        q = tf.matmul(q, self.target_weights[len(self.weights)]) + self.target_biases[len(self.weights)]
        return q

    def create_training_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.a = tf.placeholder(tf.float32, [None, self.output_size])
        self.q_value = self.feed_forward(self.x)
        self.q_value_target = self.feed_forward_target(self.x)
        self.output = tf.reduce_sum(tf.multiply(self.q_value, self.a), reduction_indices=1)
        self.action = tf.argmax(self.q_value, 1)
        self.loss = tf.reduce_mean(tf.square(self.output - self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def append_to_memory(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.output_size)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.popleft()
        if len(self.memory) > self.batch_size:
            self.train()

    def train(self):
        sample = random.sample(self.memory, self.batch_size)
        train_x = [i[0] for i in sample]
        action = [i[1] for i in sample]
        reward = [i[2] for i in sample]
        next_state = [i[3] for i in sample]
        train_y = []
        q_next = self.sess.run(self.q_value_target, feed_dict={self.x: np.array(next_state)})
        for i in range(len(reward)):
            if sample[i][4]:
                train_y.append(reward[i])
            else:
                train_y.append(reward[i] + self.gamma * np.max(q_next[i]))
        train_y = np.array(train_y)
        train_x = np.array(train_x)
        action = np.array(action)
        self.dropout = 1.0
        self.sess.run(self.optimizer, feed_dict={self.x: train_x, self.y: train_y, self.a: action})

    def copy_variables(self):
        for i in range(1, len(self.weights) + 1, 1):
            self.sess.run(self.target_weights[i].assign(self.weights[i]))
            self.sess.run(self.target_biases[i].assign(self.biases[i]))


def main():
    obj = dqn()
    for e in range(obj.episodes):
        p = env.reset()
        for i in range(500):
            obj.step += 1
            q1, ac = obj.sess.run([obj.q_value, obj.action], feed_dict={obj.x: np.array([p])})
            ac = ac[0]
            if np.random.rand() < obj.epsilon:
                ac = random.randint(0, obj.output_size - 1)
                obj.epsilon = obj.final_epsilon + (obj.initial_epsilon - obj.final_epsilon) * np.exp(
                    -obj.lambda1 * obj.step)
            obs, rew, done, _ = env.step(ac)
            obj.append_to_memory(p, ac, rew, obs, done)
            p = obs
            if done:
                break
            if obj.step % 1000 == 0:
                obj.copy_variables()

        if e % 100 == 0:
            print("episodes {0} completed".format(e), )
            av = []
            for f in range(10):
                p = env.reset()
                r = 0
                for i in range(200):
                    obj.dropout = 1.0
                    ac = obj.sess.run(obj.action, feed_dict={obj.x: np.array([p])})[0]
                    p, rew, done, _ = env.step(ac)
                    r += rew
                    if done:
                        break
                av.append(r)
            print("average score is {0}".format(np.average(np.array(av))))


if __name__ == '__main__':
    main()
