from spinup import sac,sac_core
from spinup import td3,td3_core
from spinup import ddpg,ddpg_core
from spinup import ppo,ppo_core
from spinup import sac_tf2,sac_core_tf2
from spinup import td3_tf2,td3_core_tf2

from spinup.utils.test_policy import run_policy,load_policy_and_env
import os
from utils import return_env
import torch

path=os.getcwd()+'/assets'
#In command line, run the following command lines with the "Experiment name" replaced with the name of the experiment folder
#To run experiment:
# run testing FC_gallop_speed5 --algo SAC
# run testing FC_gallop_speed5 --algo TD3

#To test:
# test testing FC_gallop_speed5

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    ### Common parameters###
    parser.add_argument('run_type', type=str, choices=('run, test'))
    parser.add_argument('exp_name', type=str)
    parser.add_argument('env_type', type=str,choices=['VMP','MC','P','L','BP','CP',
                                                      'FC','FC_gallop','FC_trot',
                                                      'FC_gallop_speed5','FC_trot_speed5',
                                                      'FC_gallop_minSpring','FC_gallop_maxSpring'])

    parser.add_argument('--algo_type', type=str,default='model_free', choices=['model_free','PID'])
    parser.add_argument('--batch_size', type=int,default=256)#256

    ### Common parameters###

    ### MODEL FREE parameters###
    parser.add_argument('--algo', type=str, default='sac', choices=('sac,SAC, td3,TD3,ppo,PPO,ddpg,DDPG,'
                                                                    'sac_tf2,tf3_tf2'))
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--gpu_choice', type=int, default=0)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int,default=100)
    parser.add_argument('--itr', type=int, default=-1)

    ### MODEL FREE parameters###

    args = parser.parse_args()
    if "tf2" not in args.algo:
        torch.set_num_threads(args.cpu)
        try:
            torch.cuda.set_device(args.gpu_choice)
        except:
            print("No GPU available")

        framework="pytorch"
    else:
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_choice)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        framework="tf2"

    save_folder=os.path.join(os.getcwd(),'data',args.exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logger_kwargs = dict(output_dir=save_folder, exp_name=args.exp_name)

    VPenv=return_env(args.env_type)

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

            elif args.algo == 'sac_tf2':
                sac_tf2(lambda: VPenv, actor_critic=sac_core_tf2.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
                    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_ep_len,
                    batch_size=args.batch_size, save_freq=args.save_freq,
                    logger_kwargs=logger_kwargs)

            elif args.algo == 'td3_tf2':
                td3_tf2(lambda: VPenv, actor_critic=td3_core_tf2.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
                    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_ep_len,
                    batch_size=args.batch_size, save_freq=args.save_freq,
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
                                                  deterministic=True,
                                                backend=framework)
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


