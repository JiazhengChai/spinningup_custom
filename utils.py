import gym
from sys import platform
import os

from assets.vertical_mvt_pendulum import VerticalMvtPendulumEnv
from assets.full_cheetah import FullCheetahEnv

xml_path=os.path.join(os.getcwd(),'assets','xml')

def return_env(choice,**kwargs):
    if choice == 'VMP':
        VPenv = VerticalMvtPendulumEnv(type='vertical_mvt_pendulum.xml', path=xml_path)
    elif choice == 'MC':
        VPenv = gym.make('MountainCarContinuous-v0')
    elif choice == 'P':
        VPenv = gym.make('Pendulum-v0')
    elif choice == 'L':
        VPenv = gym.make('LunarLanderContinuous-v2')
    elif choice == 'BP':
        VPenv = gym.make('BipedalWalker-v3')
    elif choice == 'CP':
        VPenv = gym.make('CartPole-v0')

    elif choice == 'FC':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv3.xml',path=xml_path,walkstyle='',speed=3)
    elif choice == 'FC_gallop':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv3.xml',path=xml_path,walkstyle='gallop',speed=3)
    elif choice == 'FC_trot':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv3.xml',path=xml_path,walkstyle='trot',speed=3)

    elif choice == 'FC_gallop_speed5':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv3.xml', path=xml_path, walkstyle='gallop',speed=5)
    elif choice == 'FC_trot_speed5':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv3.xml',path=xml_path,walkstyle='trot',speed=5)


    elif choice == 'FC_gallop_minSpring':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv6.xml', path=xml_path, walkstyle='gallop',speed=3)
    elif choice == 'FC_gallop_maxSpring':
        VPenv = FullCheetahEnv(xml_file='full_cheetah_heavyv7.xml', path=xml_path, walkstyle='gallop',speed=3)


    return VPenv