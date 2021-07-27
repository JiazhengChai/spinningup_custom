import time
import joblib
import os
import os.path as osp
import torch
from spinup import EpochLogger
import tensorflow as tf

def load_policy_and_env(fpath, itr='last', deterministic=False,backend = 'pytorch'):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save

    if backend!="tf2":
        # handle which epoch to load from
        if itr=='last':
            # check filenames for epoch (AKA iteration) numbers, find maximum value


            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

            itr = '%d'%max(saves) if len(saves) > 0 else ''

        else:
            assert isinstance(itr, int), \
                "Bad value provided for itr (needs to be int or 'last')."
            itr = '%d'%itr

        # load the get_action function
        get_action = load_pytorch_policy(fpath, itr, deterministic)
    else:

        # handle which epoch to load from
        if itr == 'last':
            saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x) > 11]
            itr = '%d' % max(saves) if len(saves) > 0 else ''
        else:
            itr = '%d' % itr

        # load the things!
        sess = tf.compat.v1.Session()
        model = restore_tf_graph(sess, osp.join(fpath, 'simple_save' + itr))

        # get the correct op for executing actions
        if deterministic and 'mu' in model.keys():
            # 'deterministic' is only a valid option for SAC policies
            print('Using deterministic action op.')
            action_op = model['mu']
        else:
            print('Using default action op.')
            action_op = model['pi']

        # make function for producing an action given a single state
        get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action

def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs'
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``.
    """
    tf.compat.v1.saved_model.loader.load(
                sess,
                [tf.compat.v1.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tf.compat.v1.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))