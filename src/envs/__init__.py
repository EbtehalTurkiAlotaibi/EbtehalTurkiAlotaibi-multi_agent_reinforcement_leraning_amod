from functools import partial
#from smac.env import MultiAgentEnv, StarCraft2Env
from .driver import Environment
from .multiagentenv import MultiAgentEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["driver"] = partial(env_fn, env=Environment)
#REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

