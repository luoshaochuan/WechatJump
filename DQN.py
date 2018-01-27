import tensorflow as tf
import numpy as np
import random
from collections import deque

#超参数
FRAME_PER_ACTION=1    #每个动作采几帧
GAMMA=0.99            #折扣率
OBSERVE=100.          #训练之前的观察时间
EXPLORE=10000.       #试验数
FINAL_EPSILON=0.0     #epsilon贪心最终参数
INITIAL_EPSILON = 0.5 #epsilon贪心初始参数
REPLAY_MEMORY=50000   #回放队列大小
BATCH_SIZE=32         #minbatch大小
UPDATE_TIME=100       #更新目标Q网络的时间间隔
LEARNING_RATE=0.001   #学习率
class BrainDQN:
    def __init__(self,actions):
        #初始化回放池
        self.replayMemory = deque()
        #初始化相关参数
        self.timeStep = 0
        self.epsilon=INITIAL_EPSILON
        self.actions=actions
        #初始化Q网络
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        #初始化目标Q网络并设置目标Q网络和Q网络参数一样，TD Target主要用于缓存Q网络参数值
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        self.createTrainingMethod()
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print ("Could not find old network weights")

    def createQNetwork(self):
        #三个卷积层
        W_conv1=self.weight_variable([8,8,4,32])
        b_conv1=self.bias_variable([32])

        W_conv2=self.weight_variable([4,4,32,64])
        b_conv2=self.bias_variable([64])

        W_conv3=self.weight_variable([3,3,64,64])
        b_conv3=self.bias_variable([64])
        #两个全连接层
        W_fc1=self.weight_variable([1600,512])
        b_fc1=self.bias_variable([512])

        W_fc2=self.weight_variable([512,self.actions])
        b_fc2=self.bias_variable([self.actions])
        #输入层
        stateInput=tf.placeholder(tf.float32,[None,80,80,4])#状态为4张80*80的照片
        #隐藏层
        h_conv1=tf.nn.relu(self.conv2d(stateInput,W_conv1,4)+b_conv1)
        h_pool1=self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
        # Q value 层
        QValue = tf.matmul(h_fc1,W_fc2)+b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    #缓存Q网络
    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    #把观察的图片作为状态
    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)
    #记住输出是Q值，关键要计算出cost，里面关键是计算Q_action的值，即该state和action下的Q值。由于actionInput是
    #one hot vector的形式，因此tf.mul(self.QValue, self.actionInput)正好就是该action下的Q值。
    def createTrainingMethod(self):
        self.actionInput=tf.placeholder(tf.float32,[None,self.actions])
        self.yInput=tf.placeholder(tf.float32,[None])
        Q_Action=tf.reduce_sum(tf.multiply(self.QValue,self.actionInput),reduction_indices=1)
        self.cost=tf.reduce_mean(tf.square(self.yInput-Q_Action))
        self.trainStep=tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    #主要是从回放池里随机挑选一批数据，用TD Q网络求值，然后用于更新Q网络参数
    def trainQNetwork(self):
        #step 1:从经验池里获取随机采样
        minibatch=random.sample(self.replayMemory,BATCH_SIZE)
        state_batch=[data[0] for data in minibatch]
        action_batch=[data[1] for data in minibatch]
        reward_batch=[data[2] for data in minibatch]
        nextState_batch=[data[3]for data in minibatch]
        #step 2:计算y
        y_batch=[]
        QValue_batch=self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal=minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i]+GAMMA*np.max(QValue_batch[i]))
        self.trainStep.run(feed_dict={
            self.yInput:y_batch,
            self.actionInput:action_batch,
            self.stateInput:state_batch
            })
        #保存网络
        if(self.timeStep%10000==0):
            self.saver.save(self.session,'saved_networks/'+'network'+'-dqn',global_step=self.timeStep)
        #更新TD Q网络
        if self.timeStep%UPDATE_TIME==0:
            self.copyTargetQNetwork()
    
    def setPerception(self,nextObservation,action,reward,terminal):
        newState=np.append(self.currentState[:,:,1:],nextObservation,axis=2)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory)>REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep>OBSERVE:
            self.trainQNetwork()
        #打印信息
        state=""
        if self.timeStep<=OBSERVE:
            state="observe"
        elif self.timeStep>OBSERVE and self.timeStep<=OBSERVE+EXPLORE:
            state="explore"
        else:
            state="train"
        print ("TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

    #采用epsilon贪心获取动作
    def getAction(self):
        QValue=self.QValue.eval(feed_dict={self.stateInput:[self.currentState]})[0]#所有动作的动作值函数
        action=np.zeros(self.actions)
        action_index=0
        if self.timeStep%FRAME_PER_ACTION==0:
            if(random.random()<=self.epsilon):
                action_index=random.randrange(self.actions)
                action[action_index]=1
        else:
            action[0]=1#do noting
        #change episilon
        if self.epsilon>FINAL_EPSILON and self.timeStep>OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
        return action
    #初始化状态
    def setInitState(self,observation):
        self.currentState=np.stack((observation,observation,observation,observation),axis=2)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")