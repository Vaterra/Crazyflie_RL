# Crazyflie_RL



**Runs a training script against a fixed chaser. I recommend grabing a coffee or changing the "total_timesteps: int = 500_000" to something lower for a test run, since it takes roughly 15min to run.**
`python3 Pretraining.py`

*** Runs a demo showing how the trained agent acts against the scripted chaser
`python3 Demo_trained_E_scripted_Chaser.py ***


<ins> To run tenserboard </ins>
`tensorboard --logdir=./tb_logs_chaser`
`tensorboard --logdir=./tb_logs_evader
`
*To test agents against one another*

`python3 agent_vs_agent.py`
