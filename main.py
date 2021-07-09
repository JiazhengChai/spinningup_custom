#from vertical_mvt_pendulum import VerticalMvtPendulumEnv
import os
from spinup import sac,sac_core
from spinup import td3,td3_core
from spinup import ddpg,ddpg_core
from spinup import ppo,ppo_core
from spinup.utils.test_policy import run_policy,load_policy_and_env
import gym
import os
from vertical_mvt_pendulum import VerticalMvtPendulumEnv
import torch

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

path=os.getcwd()+'/assets'
#In command line, run the following command lines with the "Experiment name" replaced with the name of the experiment folder
#To run experiment:
# run testing P --algo SAC
# run testing P --algo TD3

#To test:
# test testing P

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    ### Common parameters###
    parser.add_argument('run_type', type=str, choices=('run, test'))
    parser.add_argument('exp_name', type=str)
    parser.add_argument('env_type', type=str,choices=['VMP','MC','P','L','BP','CP'])
    parser.add_argument('--algo_type', type=str,default='model_free', choices=['model_free','model_based','PID'])
    parser.add_argument('--batch_size', type=int,default=256)#256

    ### Common parameters###

    ### MODEL FREE parameters###
    parser.add_argument('--algo', type=str, default='sac', choices=('sac,SAC, td3,TD3,ppo,PPO,ddpg,DDPG'))
    parser.add_argument('--hid', type=int, default=100)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int,default=1)
    parser.add_argument('--itr', type=int, default=-1)

    ### MODEL FREE parameters###

    args = parser.parse_args()


    save_folder=os.path.join(os.getcwd(),'data',args.exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logger_kwargs = dict(output_dir=save_folder, exp_name=args.exp_name)

    if args.env_type=='VMP':
        VPenv = VerticalMvtPendulumEnv(type='vertical_mvt_pendulum.xml', path=path)
    elif args.env_type=='MC':
        VPenv = gym.make('MountainCarContinuous-v0')
    elif args.env_type == 'P':
        VPenv = gym.make('Pendulum-v0')
    elif args.env_type == 'L':
        VPenv = gym.make('LunarLanderContinuous-v2')
    elif args.env_type == 'BP':
        VPenv = gym.make('BipedalWalker-v3')
    elif args.env_type == 'CP':
        VPenv = gym.make('CartPole-v0')

    if args.algo_type=='model_free':
        if args.run_type=='run':
            if args.algo=='sac' or args.algo=='SAC':

                sac(lambda : VPenv, actor_critic=sac_core.MLPActorCritic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
                    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,max_ep_len=args.max_ep_len,
                    batch_size=args.batch_size,save_freq=args.save_freq,
                    logger_kwargs=logger_kwargs)

            elif args.algo=='td3' or args.algo=='TD3':
                td3(lambda : VPenv, actor_critic=td3_core.MLPActorCritic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
                    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,max_ep_len=args.max_ep_len,
                    batch_size=args.batch_size,save_freq=args.save_freq,
                    logger_kwargs=logger_kwargs)


            elif args.algo == 'ppo' or args.algo == 'PPO':
                ppo(lambda: VPenv, actor_critic=ppo_core.MLPActorCritic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    seed=args.seed, steps_per_epoch=args.steps,
                    epochs=args.epochs, max_ep_len=args.max_ep_len,
                    save_freq=args.save_freq,
                    logger_kwargs=logger_kwargs)

            elif args.algo=='ddpg' or args.algo=='DDPG':
                ddpg(lambda: VPenv, actor_critic=ddpg_core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), seed=args.seed,
                     steps_per_epoch=args.steps, epochs=args.epochs, batch_size=args.batch_size,
                    logger_kwargs=logger_kwargs, save_freq=args.save_freq)


        elif args.run_type=='test':
            load_path=os.path.join(os.getcwd(),'data',args.exp_name)
            _, get_action = load_policy_and_env(load_path,
                                                  args.itr if args.itr >= 0 else 'last',
                                                  True)
            run_policy(VPenv, get_action)

    elif args.algo_type == 'PID':
        assert args.env_type=="P"

        from PID import PID
        import numpy as np
        import matplotlib.pyplot as plt


        for p in [0]:#,0.1,0.5,1,1.5,2,4
            feedback_list=[]
            PID_controller = PID(P=4, I=1, D=2,delta_time=0.01,target_pos=0)#4 0.1 1.5
            VPenv.reset()
            observation, reward, done, info=VPenv.step([np.random.uniform(-2,2)])
            for step in range(1000):
                VPenv.render()
                action,feedback_val=PID_controller.update(np.arcsin(observation[1]))
                feedback_list.append(feedback_val)
                observation, reward, done, info=VPenv.step([action])

            plt.plot(feedback_list,label=p)

        plt.legend()
        plt.show()


