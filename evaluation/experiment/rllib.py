import logging
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from env import Environment
import config
import utils
import warnings
warnings.filterwarnings("ignore")


def rllib(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
):
    # Set up environment and load checkpoint
    env = Environment(
        scheduler_name=scheduler_name,
        algo_name=algo_name,
        env_name=env_name,
        target_reward=config.envs[env_name]["max_reward"],
        budget=config.envs[env_name]["budget"],
        stop_min_round=config.stop_min_round,
        stop_max_round=config.stop_max_round,
        stop_num_results=config.stop_num_results,
        stop_cv=config.stop_cv,
        stop_grace_period=config.stop_grace_period,
        is_serverless=is_serverless,
        async_learn=False,
        use_serverless=False,
    )

    # Start training
    state, mask, info = env.reset()

    csv_round = [
        [
            "round_id",
            "duration",
            "num_rollout_workers",
            "num_envs_per_worker",
            "episodes_this_iter",
            "learner_time", 
            "actor_time",
            "eval_reward_max",
            "eval_reward_mean",
            "eval_reward_min",
            "eval_reward_std",
            "staleness_max",
            "staleness_mean",
            "staleness_min",
            "staleness_std",
            "learner_loss",
            "cost",
            "serverless_bound",
        ]
    ]

    round_id = 1

    action = {}
    action["num_rollout_workers"] = config.num_rollout_workers
    action["num_envs_per_worker"] = config.num_envs_per_worker

    csv_timeline = [
        ["Wall clock time (s)", "Episodic reward"],
        [0, 0],
    ]

    wall_clock_time = 0

    round_done = False
    while round_done is False:
        next_state, next_mask, reward, done, next_info = env.step(
            round_id=round_id,
            action=action,
            serverless_bound=114514
        )

        save_checkpoint = False

        # Save checkpoints if boost
        if save_checkpoint:
            ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, scheduler_name, env_name, algo_name, round_id)
            env.save(ckpt_path)
            json_data = {
                "round_id": round_id,
                "eval_reward_mean": next_info["eval_reward_mean"],
            }
            json_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, scheduler_name, env_name, algo_name, round_id)
            utils.json_save(json_data, json_path)
        
        staleness_max = next_info["grad_staleness_max"]
        staleness_mean = next_info["grad_staleness_mean"]
        staleness_min = next_info["grad_staleness_min"]
        staleness_std = next_info["grad_staleness_std"]
        
        csv_round.append(
            [
                round_id,
                next_info["duration"],
                action["num_rollout_workers"],
                action["num_envs_per_worker"],
                next_info["episodes_this_iter"],
                next_info["learner_time"],
                next_info["actor_time"],
                next_info["eval_reward_max"],
                next_info["eval_reward_mean"],
                next_info["eval_reward_min"],
                next_info["eval_reward_std"],
                staleness_max,
                staleness_mean,
                staleness_min,
                staleness_std,
                next_info["learner_loss"],
                next_info["cost"],
                0,
            ]
        )

        wall_clock_time = wall_clock_time + next_info["duration"]
        if next_info["episode_reward"]:
            for r in next_info["episode_reward"]:
                csv_timeline.append([wall_clock_time, r])
        else:
            csv_timeline.append([wall_clock_time, 0])

        print("")
        print("******************")
        print("******************")
        print("******************")
        print("")
        print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
        print("round_id: {}".format(next_info["round_id"]))
        print("duration: {}".format(next_info["duration"]))
        print("action: {}".format(action))
        print("eval_reward_mean: {}".format(next_info["eval_reward_mean"]))
        print("cost: {}".format(next_info["cost"]))

        if done:
            utils.export_csv(
                scheduler_name=scheduler_name,
                env_name=env_name, 
                algo_name=algo_name, 
                csv_name=f"info",
                csv_file=csv_round
            )
            utils.export_csv(
                scheduler_name=scheduler_name,
                env_name=env_name,
                algo_name=algo_name,
                csv_name=f"timeline",
                csv_file=csv_timeline,
            )

            env.stop_trainer()
            round_done = True

        state = next_state
        mask = next_mask
        info = next_info
        round_id = round_id + 1

    
if __name__ == '__main__':
    args = utils.get_parse()

    scheduler_name = "rllib"
    is_serverless = False

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    
    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.INFO,
        include_dashboard=False,
    )

    rllib(
        scheduler_name=scheduler_name,
        is_serverless=is_serverless,
        algo_name=args.algo_name,
        env_name=args.env_name,
    )

    ray.shutdown()
    
    print("")
    print("**********")
    print("**********")
    print("**********")