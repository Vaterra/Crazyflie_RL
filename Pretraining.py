from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from evader_env import EvaderPretrainEnv
from chaser_env import ChaserPretrainEnv


def pretrain_evader(
    total_timesteps: int = 200_000,
    model_path: str = "evader_pretrain_ppo",
):
    env = EvaderPretrainEnv()
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        #device="cpu",
        tensorboard_log="./tb_logs_evader",
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="evader_pretrain",
    )

    model.save(model_path)
    print(f"[Evader] Saved model to: {model_path}.zip")
    return model

def pretrain_chaser(
    total_timesteps: int = 200_000,
    model_path: str = "chaser_pretrain_ppo",
):
    env = ChaserPretrainEnv()
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        #device="gpu",
        tensorboard_log="./tb_logs_chaser",
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="chaser_pretrain",
    )

    model.save(model_path)
    print(f"[Chaser] Saved model to: {model_path}.zip")
    return model


if __name__ == "__main__":
    pretrain_evader()