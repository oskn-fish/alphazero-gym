# What is this
This repo contains multiple, self-contained implementations of AlphaZero. 
They are based on this repo https://github.com/tmoer/alphazero_singleplayer
as well as the following paper [A0C: Alpha Zero in Continuous Action Space](https://arxiv.org/pdf/1805.09613.pdf)

## master branch

## tf branch
Base implementation.  

## TODOS
- [x] Verify that this works with tensorflow (in a virtualenv).
- [x] Port models to PyTorch and test.
- [x] Implement tensorboard visualization.
- [x] Implement Agent abstraction.
- [x] Create branch for current base version.
- [x] Create proper policy methods for MCTS.
- [x] restructure project.
- [x] Implement model saving and loading.
- [x] Implement more detailed logging.
- [x] Decouple node and mcts.
- [x] Implement Simulation policy
- [ ] Implement MCTS progressive widening.
- [ ] Implement A0C.
- [ ] Use Pendulum to test (might need some custom wrappers).
- [ ] Enable GPU training.


## Possible Enhancements to try
- [ ] Value loss scaling.
- [ ] Test a prioritized replay buffer.
- [ ] Warm starting.
- [ ] Playout cap randomization.
- [ ] RAD style data augmentations.
