from spinup.utils.plot import my_make_plots,my_return_plots
import os
import argparse
import matplotlib.pyplot as plt
import pandas

DIAGRAM_NAME='comparisons'

data_path=os.path.join(os.getcwd(),'data')
save_path=os.path.join(os.getcwd(),'plot',DIAGRAM_NAME)

path_to_trained_models=[
                        os.path.join(data_path,'cptest'),
                        os.path.join(data_path,'cptest2')
                        ]

parser = argparse.ArgumentParser()
parser.add_argument('--legend', '-l', nargs='*')
parser.add_argument('--xaxis', '-x', default='Epoch')#TotalEnvInteracts
parser.add_argument('--value', '-y', default='AverageTestEpRet', nargs='*')# EpRet AverageTestEpRet
parser.add_argument('--count', action='store_true')
parser.add_argument('--smooth', '-s', type=int, default=1)
parser.add_argument('--select', nargs='*')
parser.add_argument('--exclude', nargs='*')
parser.add_argument('--est', default='mean')
args = parser.parse_args()

my_return_plots(path_to_trained_models, args.legend, args.xaxis, args.value, args.count,
           smooth=args.smooth, select=args.select, exclude=args.exclude,
           estimator=args.est,save_path=save_path)


if len(DIAGRAM_NAME)>0:
    plt.savefig(save_path)
else:
    plt.show()
