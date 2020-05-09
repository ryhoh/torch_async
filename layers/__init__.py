from .base import Rotatable, MyConv2d
from .static import SemisyncLinear, SequentialLinear
from .random_semisync import RandomSemiSyncLinear, RandomSequentialLinear, RandomSemiSyncConv2d
from .stochastic_semisync import StochasticSemiSyncLinear, StochasticSequentialLinear

# alias for old version
from .static import SemisyncLinear as OptimizedSemiSyncLinear
from .static import SequentialLinear as OptimizedContinuousLinear
