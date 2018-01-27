import numpy as np
import os
import time
import math
import random
from PIL import Image
import operator
from functools import reduce
#棋子大小参数
PIECE_BASE_HEIGHT=20
PIECE_BODY_WIDTH=70

#再玩一局
PLAY_AGAIN=Image.open('play_again.png');

#通过哈希算法比较当前图片和“再玩一局”是否一样
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)
def get_terminal(im):
    img=im.crop((200,1250,700,1400)).resize((200,60))
    diff=hist_similar(PLAY_AGAIN.histogram(),img.histogram())
    return diff>0.6

#获取屏幕截图
def pull_screenshot():
    os.system('adb shell screencap -p /sdcard/autojump.png')
    os.system('adb pull /sdcard/autojump.png .')

#传入截图大小
def set_button_position(w,h):
    """
    将 swipe 设置为 `再来一局` 按钮的位置
    """
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))    # 随机防 ban
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top


def press(press_time):
    """
    按压一定时间
    """
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    #print(cmd)
    os.system(cmd)
    return press_time

def init_state():
    pull_screenshot()
    im = Image.open('./autojump.png')
    im=im.transpose(Image.ROTATE_90).convert('L')#在BlueStack里截的图是横着的，所以我旋转了一下,并转为灰度图
    image_data=np.array(im.crop((50,450,850,1250)).resize((80,80)))
    return image_data

#action：按压时间，采用one-hot编码，action[i]=1表示按压i*100ms
#return: image_data:当前截图；reward:奖励；terminal:是否终止
def frame_step(action):
    reward=0
    terminal=False
    press_time=(action.argmax()+3)*100
        
    set_button_position(900,1600)
    press(press_time)

    #跳之后
    time.sleep(random.uniform(press_time/100,press_time/100))
    pull_screenshot()
    im = Image.open('./autojump.png')
    im=im.transpose(Image.ROTATE_90).convert('L')#在BlueStack里截的图是横着的，所以我旋转了一下,并转为灰度图
    terminal=get_terminal(im)
    if terminal:
        press(200)#重新开始
        reward=-1
    else:
        reward=1
    image_data=np.array(im.crop((50,450,850,1250)).resize((80,80)))
    im.close()
    print(reward,terminal)
    return image_data,reward,terminal
