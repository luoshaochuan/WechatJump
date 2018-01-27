import wrapped_wechat_jump as game
from DQN import BrainDQN
import numpy as np

def playWeChatJump():
    actions = 10
    brain = BrainDQN(actions)#action采用one-hot编码
    observation0=game.init_state()
    brain.setInitState(observation0)
    while 1!= 0:
        action = brain.getAction()
        nextObservation,reward,terminal = game.frame_step(action)
        nextObservation = np.reshape(nextObservation,(80,80,1))
        brain.setPerception(nextObservation,action,reward,terminal)

def main():
	playWeChatJump()

if __name__ == '__main__':
	main()