from absl import flags
from mafis.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "mamujoco": MAMuJoCoLogger,
}
