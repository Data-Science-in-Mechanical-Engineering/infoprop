# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.infoprop_dyna as infoprop_dyna

import mbrl.util.env



@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    algo_name = cfg.algorithm.name
    task = cfg.overrides.env
    print("Running "+algo_name+" for the "+task+" task.")

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    test_env = None
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "infoprop_dyna":
        if not test_env:
            test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return infoprop_dyna.train(env, test_env, term_fn, cfg)

if __name__ == "__main__":
    run()
