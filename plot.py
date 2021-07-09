from spinup.utils.plot import my_make_plots,my_return_plots
import os
import argparse
import matplotlib.pyplot as plt
import pandas

DIAGRAM_NAME='comparisons'

data_path=os.path.join(os.getcwd(),'data')
save_path=os.path.join(os.getcwd(),'plot',DIAGRAM_NAME)
'''path_to_trained_models=[
                        os.path.join(data_path,'VMP_sac'),
                        os.path.join(data_path,'VMP_TD3'),
                        ]

path_to_model_based_model=[
                            os.path.join(data_path,'VMP_model_based'),
                            ]'''

path_to_trained_models=[
                        os.path.join(data_path,'modif_SAC'),
                        os.path.join(data_path,'real_SAC'),
                        ]

path_to_trained_models=[
                        os.path.join(data_path, 'modif_SAC'),
                        os.path.join(data_path, 'real_SAC'),
                        os.path.join(data_path,'modif_TD3'),
                        os.path.join(data_path,'real_TD3'),
                        os.path.join(data_path,'real_PPO'),
                        os.path.join(data_path,'real_TRPO'),
                        os.path.join(data_path, 'modif_DDPG'),
                        os.path.join(data_path, 'real_DDPG'),
                        ]
path_to_model_based_model=[]

parser = argparse.ArgumentParser()
parser.add_argument('logdir', nargs='*')
parser.add_argument('--legend', '-l', nargs='*')
parser.add_argument('--xaxis', '-x', default='Epoch')#TotalEnvInteracts
parser.add_argument('--value', '-y', default='Performance', nargs='*')#Performance EpRet AverageTestEpRet
parser.add_argument('--count', action='store_true')
parser.add_argument('--smooth', '-s', type=int, default=1)
parser.add_argument('--select', nargs='*')
parser.add_argument('--exclude', nargs='*')
parser.add_argument('--est', default='mean')
args = parser.parse_args()

if len(args.logdir)!=0:
    my_return_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est,save_path=save_path)

else:

    my_return_plots(path_to_trained_models, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est,save_path=save_path)

if len(path_to_model_based_model)>0:
    for model in path_to_model_based_model:
        csv_file=os.path.join(model,'log.csv')

        csv_table=pandas.read_csv(csv_file)

        returns=csv_table['ReturnAvg']

        plt.plot(range(0,55000,5000),returns[0:11],color='red',label='model_based')
        plt.legend(loc='best')


plt.show()

'''if len(DIAGRAM_NAME)>0:
    plt.savefig(save_path)
else:
    plt.show()'''
