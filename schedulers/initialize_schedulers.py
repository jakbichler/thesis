from schedulers.greedy_instantaneous_scheduler import GreedyInstantaneousScheduler
from schedulers.random_bipartite_matching_scheduler import RandomBipartiteMatchingScheduler
from schedulers.sadcher import SadcherScheduler, StochasticILSadcherScheduler
from schedulers.sadcherRL import RLSadcherScheduler


def create_scheduler(
    name: str,
    checkpoint_path=None,
    duration_normalization=100,
    location_normalization=100,
    debugging=False,
    stddev=1,
):
    if name == "greedy":
        return GreedyInstantaneousScheduler()
    elif name == "random_bipartite":
        return RandomBipartiteMatchingScheduler()
    elif name == "sadcher":
        return SadcherScheduler(
            debugging=debugging,
            checkpoint_path=checkpoint_path,
            duration_normalization=duration_normalization,
            location_normalization=location_normalization,
        )
    elif name in ["rl_sadcher", "rl_sadcher_sampling"]:
        return RLSadcherScheduler(
            debugging=debugging,
            checkpoint_path=checkpoint_path,
            duration_normalization=duration_normalization,
            location_normalization=location_normalization,
        )
    elif name == "stochastic_IL_sadcher":
        return StochasticILSadcherScheduler(
            debugging=debugging,
            checkpoint_path=checkpoint_path,
            duration_normalization=duration_normalization,
            location_normalization=location_normalization,
            stddev=stddev,
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'")
