# Algorithms
from spinup.algos.pytorch.ddpg.ddpg import ddpg
import spinup.algos.pytorch.ddpg.core as ddpg_core

from spinup.algos.pytorch.ppo.ppo import ppo
import spinup.algos.pytorch.ppo.core as ppo_core

from spinup.algos.pytorch.sac.sac import sac
import spinup.algos.pytorch.sac.core as sac_core

from spinup.algos.pytorch.td3.td3 import td3
import spinup.algos.pytorch.td3.core as td3_core


# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__