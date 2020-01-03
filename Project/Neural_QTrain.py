import gym

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 1.0 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini_batch

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# TODO: Define Network Graph
HIDDEN_NODES = 100
replay_buffer = []
##### build my network
def my_network(state_dim, action_dim, hidden_nodes = HIDDEN_NODES):
    ## state_in => hidden_layer => Q value layer
    w1 = tf.Variable(tf.random_normal([state_dim, hidden_nodes], stddev = 0.1))
    b1 = tf.Variable(tf.zeros([hidden_nodes]))

    w2 = tf.Variable(tf.random_normal([hidden_nodes, action_dim], stddev = 0.1))
    b2 = tf.Variable(tf.zeros([action_dim]))

    # hidden layer
    h_layer = tf.nn.relu(tf.matmul(state_in, w1) + b1)

    # Q Value layer
    q_values = tf.matmul(h_layer, w2) + b2

    q_action = \
        tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices = 1)

    # loss function and optimizer
    network_loss = tf.reduce_sum(tf.square(target_in - q_action))

    network_optimizer = tf.train.AdamOptimizer().minimize(network_loss)

    train_loss_summary_op = tf.summary.scalar("TrainingLoss", network_loss)

    return state_in, action_in, target_in, q_values, q_action, \
           network_loss, network_optimizer, train_loss_summary_op

return_values = my_network(STATE_DIM, ACTION_DIM)

state_in = return_values[0]
action_in = return_values[1]
target_in = return_values[2]

# TODO: Network outputs
q_values = return_values[3]
q_action = return_values[4]

# TODO: Loss/Optimizer Definition
loss = return_values[5]
optimizer = return_values[6]

# train_loss_summary_op
train_loss_summary_op = return_values[7]

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

replay_buffer = [] 	# store training information

def perceive(state, action, reward, next_state, done, replay_buffer):
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    replay_buffer.append((state, one_hot_action, reward, next_state, done))
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer = replay_buffer[1:]	# get rid of the first training information

    if len(replay_buffer) > BATCH_SIZE:		
        return True			# need to train

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        action_index = list(action).index(1)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated

        ###### PERCEIVE FUNCTION #######
        result = perceive(state, action_index, reward, next_state, done, replay_buffer)
        if result: # result == True
            # need to train

            # Step 1: randomly pick some mini_batch information from replay_buffer
            mini_batch = []
            L = replay_buffer[:]
            for _ in range(BATCH_SIZE):
                index = random.randrange(len(L))
                mini_batch.append(L.pop(index))

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            for e in mini_batch:
                state_batch.append(e[0])
                action_batch.append(e[1])
                reward_batch.append(e[2])
                next_state_batch.append(e[3])

            # Step 2: calculate target q
            target_batch = []
            q_value_batch = q_values.eval(feed_dict={state_in: next_state_batch})
            for i in range(0, BATCH_SIZE):
                batch_done = mini_batch[i][4]
                if batch_done:		# terminate, no need to update next state's information
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))
                    ## update q value based on Bellman function, according to hint 1

            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
