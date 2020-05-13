# RTFM

This repository contains code for [RTFM: Generalizing to Novel Environment via Reading](https://arxiv.org/abs/1910.08210).
In particular, it contains

- RTFM, a suite of procedurally generated environments that require jointly reasoning over a language goal, environment observations, and a document describing high-level environment dynamics.

- txt2pi, a model that beats existing state-of-the-art models on RTFM.

## Citation
If you use this work, please cite:

```bib
@inproceedings{
  Zhong2020RTFM,
  title={RTFM: Generalising to New Environment Dynamics via Reading},
  author={Victor Zhong and Tim Rockt\"{a}schel and Edward Grefenstette},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/abs/1910.08210}
}
```

## Setup

First, set up RTFM as follows:

```
pip install -e .
```

## Play through

You can play the environments manually to see what the generated game instances are like via:

```
python play_gym.py -c --env groups_nl --shuffle_wiki  # this is the full RTFM tasks described in the paper
python play_gym.py -c --env rock_paper_scissors --shuffle_wiki  # this is the rock paper scissors task described in the paper
```

Note that the agent doesn't actually see textual icons such as `!`, `@`, and `?`.
Instead, it only sees the text description (e.g. `fire goblin`).
For a list of all games, including simpler variants of RTFM, please see `rtfm/tasks/__init__.py`.


## Training agents

Training is done by running `run_exp.py`, which is forked off an older implementation of [Torchbeast](https://github.com/facebookresearch/torchbeast).
Note that due to the distriuted nature of Torchbeast, the runs are stochastic.
We use multiple runs to produce the plots and analysis, however for exposition purposes the command for only one run is shown.
Our experiments were run on a slurm cluster using machines with 30 CPUs, 1 GPU, and 32GB of RAM.
Note that if you want to run multiple jobs and have them log to different directories, use `--prefix <job_id>`.
The output are located in the `checkpoints` directory by default.
For convenience, you can use `python termplot.py checkpoints/<savedir>` to visualize the training progress.


### Simplest RTFM

```
export CONST=--demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0007 --total_frames=100000000 --height 6 --width 6
python run_exp.py --env rtfm:groups_simple_stationary-v0 --model paper_conv $CONST
python run_exp.py --env rtfm:groups_simple_stationary-v0 --model paper_film $CONST
python run_exp.py --env rtfm:groups_simple_stationary-v0 --model paper_txt2pi $CONST
```

### Curriculum learning

Stage 0: random initialization:

```
export CONST=--model paper_txt2pi --demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0007 --entropy_cost 5e-3 --total_frames=100000000 --height 6 --width 6

# simplest 6x6
python run_exp.py --env rtfm:groups_simple_stationary-v0 $CONST  
# +nl
python run_exp.py --env rtfm:groups_simple_stationary_nl-v0 $CONST
# +dyna
python run_exp.py --env rtfm:groups_simple-v0 $CONST
# +groups
python run_exp.py --env rtfm:groups_stationary-v0 $CONST
# +groups,nl
python run_exp.py --env rtfm:groups_stationary_nl-v0 $CONST
# +dyna,nl
python run_exp.py --env rtfm:groups_simple_nl-v0 $CONST
# +dyna,groups
python run_exp.py --env rtfm:groups-v0 $CONST
# +dyna,groups,nl
python run_exp.py --env rtfm:groups_nl-v0 $CONST
```

Stage 1: resume from simplest 6x6

```
export RESUME=--resume $PWD/exp/<SAVE>/model.tar

# +nl
python run_exp.py --env rtfm:groups_simple_stationary_nl-v0 $CONST $RESUME
# +dyna
python run_exp.py --env rtfm:groups_simple-v0 $CONST $RESUME
# +groups
python run_exp.py --env rtfm:groups_stationary-v0 $CONST $RESUME
# +groups,nl
python run_exp.py --env rtfm:groups_stationary_nl-v0 $CONST $RESUME
# +dyna,nl
python run_exp.py --env rtfm:groups_simple_nl-v0 $CONST $RESUME
# +dyna,groups
python run_exp.py --env rtfm:groups-v0 $CONST $RESUME
# +dyna,groups,nl
python run_exp.py --env rtfm:groups_nl-v0 $CONST $RESUME
```

Stage 2: resume from `+dyna`

```
export RESUME=--resume $PWD/exp/<SAVE>/model.tar

# +dyna,nl
python run_exp.py --env rtfm:groups_simple_nl-v0 $CONST $RESUME
# +dyna,groups
python run_exp.py --env rtfm:groups-v0 $CONST $RESUME
# +dyna,groups,nl
python run_exp.py --env rtfm:groups_nl-v0 $CONST $RESUME
```

Stage 3: resume from `+dyna,group`

```
export RESUME=--resume $PWD/exp/<SAVE>/model.tar
export CONST=--model paper_txt2pi --demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0001 --entropy_cost 5e-3 --total_frames=100000000 --height 6 --width 6

# +dyna,groups,nl
python run_exp.py --env rtfm:groups_nl-v0 $CONST $RESUME
```


## Trained binaries

We provide some trained binaries for illustration purposes.
To get them, please [download the content from here](https://drive.google.com/file/d/1jh4uv_oN3lckUzH60ryqFLdgOmMU4wWl/view?usp=sharing) and extract it into the `./checkpoints` directory.
You can evaluate the model as follows:


```
python run_exp.py --mode test --env rtfm:groups_nl-v0 --model paper_txt2pi --resume checkpoints/groups_nl:paper_txt2pi:yeswiki:default/model.tar
```

To to see playthrough trajectories, you need to change `--mode test` to `--mode test_render`.



## Rock paper scissors

```
python run_exp.py --env rtfm:rock_paper_scissors-v0 --model paper_conv --demb 10 --drnn 100 --drep 300
python run_exp.py --env rtfm:rock_paper_scissors-v0 --model paper_film --demb 10 --drnn 100 --drep 300
python run_exp.py --env rtfm:rock_paper_scissors-v0 --model paper_txt2pi --demb 10 --drnn 100 --drep 300
```

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
RTFM is Attribution-NonCommercial 4.0 International licensed, as found in the LICENSE file.
