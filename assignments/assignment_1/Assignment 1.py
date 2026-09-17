# WE USE THIS 
###############IIIIIIIIII


# everything imported from the template

# Standard library
import random
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]


# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
# ============================================================================ #


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


#Make population (random generate, size 100 for now, set it so population will be same for independent runs)


#Point mutation --> Anne, Alecsia


#Sub-tree mutation --> Carolien, Sandra


#Crossover


#Selection


#Run for point 5x, run for sub-tree 5x, run for either third mutation or both point and sub-tree mutation
#make graphs




















# ============================================================================ #
# 1. IMPORTS
# ============================================================================ #
import random
from pathlib import Path
from typing import Literal

import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# ARIEL EC Engine Imports
from ariel.ec import EA, EAOperation, Individual, Population


# ============================================================================ #
# 2. EXPERIMENT CONSTANTS & SETUP
# ============================================================================ #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

TARGET_DIR: Path = HERE / "target_bodies"  
NUM_OF_MODULES: int = 20  
GENOTYPE: Literal["nde", "tree"] = "tree" 
MODE: Literal["launcher", "video", "frame", "none"] = "frame" 
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

# --- EXPERIMENTAL BATCH CONSTANTS ---
EXPERIMENTS = ["point_mutation", "subtree_mutation", "random_baseline"]
SEEDS = [42, 101, 2026, 888, 7] 
NUM_GENERATIONS = 100
POP_SIZE = 100


# ============================================================================ #
# 3. HELPER FUNCTIONS
# ============================================================================ #
def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]

def fitness_function(body: nx.DiGraph, targets: list[nx.DiGraph]) -> float:
    return mean_plus_std_tree_edit_distance(body, targets)

def show_body(body: nx.DiGraph, mode="frame", file_name="body") -> None:
    if mode == "none":
        return

    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            viewer.launch(model=model, data=data)
        case "frame":
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


# ============================================================================ #
# 4. CUSTOM EA OPERATIONS (YOUR TEAM'S WORK GOES HERE)
# ============================================================================ #