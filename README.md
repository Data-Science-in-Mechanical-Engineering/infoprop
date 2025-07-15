# infoprop submission code
This project has the submission code for the paper "On Rollouts in Model-Based Reinforcement Learning" [arxiv](https://arxiv.org/abs/2501.16918). This project is an adapted version of [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) by facebook research. Only Infoprop Dyna is implemented here. To run the other algorithms used for comparison, use [MACURA](https://github.com/Data-Science-in-Mechanical-Engineering/macura). 
## Installation Instructions
First, change directory to where the project is located:
```bash
cd <root_directory_of_the_project>
```
To run the code, you will need anaconda installed on your system. First, to install the conda environment run:
```bash
conda env create -f environment.yaml
```
This will create a conda environment named infoprop. To activate the environment run:
```bash
conda activate infoprop
```
Next, you will need to install the mbrl package. To do this run:
```bash
pip install -e .
```
To see if everything works as expected, run the following command in terminal:
```bash
python -m mbrl.examples.main
```
You should see that an Infoprop Dyna experiment has started with default parameters. To terminate use <kbd>Ctrl</kbd>+<kbd>c</kbd>. We use hydra for configuration management, for information visit [hydra-docs](https://hydra.cc/docs/intro/). In order to run an experiment on another environment, for example hopper, use:
```bash
python -m mbrl.examples.main overrides=info_hopper
```
By default, the experiments are logged in csv files within the exp folder. For advanced logging and visualization we use [weights and biases](https://wandb.ai/site)(W&B). For creating a weights and biases account, visit [W&B account creation](https://docs.wandb.ai/quickstart). For additional documentation visit [wandb docs](https://docs.wandb.ai/). In order to enable W&B logging, run the experiment with:
```bash
python -m mbrl.examples.main wandb_log=True wandb_project=Infoprop experiment=wandb_trial_run 
```
This will log the current experiment in the project "Infoprop" with the name "wandb_trial_run".
