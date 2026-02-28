# Crazyflie_RL



## Runs a training script against a fixed chaser.
## I recommend grabing a coffee or changing the "total_timesteps: int = 500_000" to something lower for a test run, since it takes roughly 15min to run
``code: python3 Pretraining.py

# Runs a demo showing how the trained agent acts against the scripted chaser
code: python3 Demo_trained_E_scripted_Chaser.py


# To run tenserboard
code: tensorboard --logdir=./tb_logs_chaser
or: tensorboard --logdir=./tb_logs_evader
