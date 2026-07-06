import os
from pathlib import Path
import numpy as np

from ..utils.myutils import load_config

ENV = load_config(os.path.join(Path(__file__).parent.parent, "config/pnc_env.yaml"))
SIM = load_config(os.path.join(Path(__file__).parent.parent, "config/simulation.yaml"))
AGN = load_config(os.path.join(Path(__file__).parent.parent, "config/agent.yaml"))
RLC = load_config(os.path.join(Path(__file__).parent.parent, "config/rl_config.yaml"))
INF = load_config(os.path.join(Path(__file__).parent.parent, "config/inference.yaml"))

### AGN
for key in AGN:
    if key not in ["reactions", "observations", "param_moduls", "actions"]:
        if isinstance(AGN[key]["reaction"], str):
            AGN[key]["reaction"] = AGN["reactions"][AGN[key]["reaction"]]
        if isinstance(AGN[key]["observation"], str):
            AGN[key]["observation"] = AGN["observations"][AGN[key]["observation"]]
        if isinstance(AGN[key]["param_modul"], str):
            AGN[key]["param_modul"] = AGN["param_moduls"][AGN[key]["param_modul"]]
        if isinstance(AGN[key]["action"], str):
            AGN[key]["action"] = AGN["actions"][AGN[key]["action"]]
        