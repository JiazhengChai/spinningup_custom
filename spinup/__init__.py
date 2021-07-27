# Algorithms
from spinup.algos.pytorch.ddpg.ddpg import ddpg
import spinup.algos.pytorch.ddpg.core as ddpg_core

from spinup.algos.pytorch.ppo.ppo import ppo
import spinup.algos.pytorch.ppo.core as ppo_core

from spinup.algos.pytorch.sac.sac import sac
import spinup.algos.pytorch.sac.core as sac_core

from spinup.algos.pytorch.td3.td3 import td3
import spinup.algos.pytorch.td3.core as td3_core

from spinup.algos.tf2.sac.sac import sac as sac_tf2
import spinup.algos.tf2.sac.core as sac_core_tf2

from spinup.algos.tf2.td3.td3 import td3 as td3_tf2
import spinup.algos.tf2.td3.core as td3_core_tf2

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
