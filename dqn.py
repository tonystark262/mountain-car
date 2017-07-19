import os
import random

import gym
import numpy as np
import tensorflow as tf
from collections import deque

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
        self.epsilon = 0.5
        self.step = 0
        self.learning_rate = 0.001
        self.dropout = 1.0
        self.lambda1 = 0.01
        self.initial_epsilon = self.epsilon
        self.final_epsilon = 0.01
        self.weights = {}
        self.biases = {}
        self.create_nn()
        self.create_training_network()
        self.memory = deque()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def create_nn(self):
        self.weights[1] = tf.Variable(tf.truncated_normal([self.input_size, 128]), name='w1')
        #'''
        self.weights[2] = tf.Variable(tf.truncated_normal([128, 256]), name='w2')
        self.weights[3] = tf.Variable(tf.truncated_normal([256, 512]), name='w3')
        self.weights[4] = tf.Variable(tf.truncated_normal([512, 256]), name='w4')
        self.weights[5] = tf.Variable(tf.truncated_normal([256, 128]), name='w5')
        #'''
        self.weights[6] = tf.Variable(tf.truncated_normal([128, self.output_size]), name='w6')
        self.biases[1] = tf.Variable(tf.truncated_normal([128]), name='b1')
        #'''
        self.biases[2] = tf.Variable(tf.truncated_normal([256]), name='b2')
        self.biases[3] = tf.Variable(tf.truncated_normal([512]), name='b3')
        self.biases[4] = tf.Variable(tf.truncated_normal([256]), name='b4')
        self.biases[5] = tf.Variable(tf.truncated_normal([128]), name='b5')
        #'''
        self.biases[6] = tf.Variable(tf.truncated_normal([self.output_size]), name='b6')

    def feed_forward(self, z):
        q = tf.nn.tanh(tf.matmul(z, self.weights[1]) + self.biases[1])
        #'''
        q = tf.nn.tanh(tf.matmul(q, self.weights[2]) + self.biases[2])
        q = tf.nn.tanh(tf.matmul(q, self.weights[3]) + self.biases[3])
        q = tf.nn.tanh(tf.matmul(q, self.weights[4]) + self.biases[4])
        q = tf.nn.tanh(tf.matmul(q, self.weights[5]) + self.biases[5])
        #'''
        q = tf.matmul(q, self.weights[6]) + self.biases[6]
        return q

    def create_training_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.a = tf.placeholder(tf.float32, [None, self.output_size])
        self.q_value = self.feed_forward(self.x)
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
        q_next = self.sess.run(self.q_value, feed_dict={self.x: np.array(next_state)})
        for i in range(len(reward)):
            if sample[i][4]:
                train_y.append(reward[i])
            else:
                train_y.append(reward[i] + self.gamma * np.max(q_next[i]))
        train_y = np.array(train_y)
        train_x = np.array(train_x)
        action = np.array(action)
        self.sess.run(self.optimizer, feed_dict={self.x: train_x, self.y: train_y, self.a: action})


def main():
    obj = dqn()
    for e in range(obj.episodes):
        p = env.reset()
        for i in range(500):
            obj.step += 1
            ac = obj.sess.run(obj.action, feed_dict={obj.x: np.array([p])})
            ac = ac[0]
            if random.random() < obj.epsilon:
                ac = random.randint(0,obj.output_size-1)
                obj.epsilon -= (obj.initial_epsilon-obj.final_epsilon) / 10000
            obs, rew, done, _ = env.step(ac)
            obj.append_to_memory(p, ac, rew, obs, done)
            p=obs
            if done:
                break
        if e % 100 == 0:
            print("episodes {0} completed".format(e), )
            av = 0
            for f in range(10):
                p = env.reset()
                for i in range(200):
                    obj.dropout = 1.0
                    ac = obj.sess.run(obj.action, feed_dict={obj.x: np.array([p])})[0]
                    p, rew, done, _ = env.step(ac)
                    av+=rew
                    if done:
                        break
            av/=10
            print("average score is {0}".format(av))


if __name__ == '__main__':
    main()
