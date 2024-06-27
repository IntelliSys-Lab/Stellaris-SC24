import numpy as np
from pprint import pprint
import torch
import redis
import multiprocessing
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.algorithms import ppo, appo, impala
import pickle
import json
import csv
import time
import collections
import config
import utils


class Environment():
    """ 
    Environment for Minions
    """

    def __init__(
        self,
        scheduler_name,
        algo_name,
        env_name,
        target_reward,
        budget,
        stop_min_round,
        stop_max_round,
        stop_num_results,
        stop_cv,
        stop_grace_period,
        is_serverless,
        async_learn,
        use_serverless,
    ):
        self.scheduler_name = scheduler_name
        self.algo_name = algo_name
        self.env_name = env_name
        self.target_reward = target_reward
        self.budget = budget
        self.stop_min_round = stop_min_round
        self.stop_max_round = stop_max_round
        self.stop_num_results = stop_num_results
        self.stop_cv = stop_cv
        self.stop_grace_period = stop_grace_period
        self.is_serverless = is_serverless
        self.async_learn = async_learn
        self.use_serverless = use_serverless

        if self.is_serverless:
            self.lambda_algo_name = self.algo_name
            self.algo_name = "{}_serverless".format(self.algo_name)

        if self.async_learn or self.is_serverless:
            self.init_redis_client()

        self.init_trainer_config()
        self.trainer = None

        self.stop_window = collections.deque(maxlen=self.stop_num_results)
        # self.reset_cost()
        self.max_eval_reward = -1e8

    def init_redis_client(self):
        self.pool = redis.ConnectionPool(
            host=config.redis_host, 
            port=config.redis_port, 
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)

    def stop_trainer(self):
        if self.trainer is not None:
            self.trainer.stop()
            self.trainer = None

    def init_trainer_config(self):
        if self.algo_name == "ppo":
            trainer_config = ppo.PPOConfig()
        elif self.algo_name == "appo":
            trainer_config = appo.APPOConfig()
        elif self.algo_name == "impala":
            trainer_config = impala.ImpalaConfig()
        elif self.algo_name == "ppo_serverless":
            trainer_config = ppo.PPOServerlessConfig()
        elif self.algo_name == "appo_serverless":
            trainer_config = appo.APPOServerlessConfig()
        elif self.algo_name == "impala_serverless":
            trainer_config = impala.ImpalaServerlessConfig()

        # Init with the same number of actors
        num_rollout_workers = config.num_rollout_workers
        num_envs_per_worker = config.num_envs_per_worker
        rollout_fragment_length = config.envs[self.env_name]['rollout_fragment_length']
        if self.algo_name in ["ppo", "ppo_serverless"]:
            train_batch_size = num_rollout_workers * num_envs_per_worker * rollout_fragment_length
        else:
            train_batch_size = num_envs_per_worker * rollout_fragment_length
        
        trainer_config = (
            trainer_config
            .framework(framework=config.framework)
            # .callbacks(callbacks_class=CustomCallbacks)
            .environment(env=self.env_name)
            .resources(
                num_gpus=config.num_gpus_for_local_worker,
                num_cpus_for_local_worker=config.num_cpus_for_local_worker,
                num_cpus_per_worker=config.num_cpus_per_worker,
                num_gpus_per_worker=config.num_gpus_per_worker,
            )
            .rollouts(
                rollout_fragment_length=rollout_fragment_length,
                num_rollout_workers=num_rollout_workers,
                num_envs_per_worker=num_envs_per_worker,
                # batch_mode="complete_episodes",
                batch_mode="truncate_episodes",
            )
            .debugging(
                log_level="ERROR",
                logger_config={"type": ray.tune.logger.NoopLogger},
                log_sys_usage=False
            ) # Disable all loggings to save time
        )

        # Configure report time to avoid learner thread dies: https://discuss.ray.io/t/impala-bugs-and-some-other-observations/9863/7
        trainer_config = trainer_config.reporting(
            min_time_s_per_iteration=config.min_time_s_per_iteration,
        )

        # Configure experimental settings
        trainer_config = trainer_config.experimental(
            _enable_new_api_stack=config._enable_new_api_stack,
        )

        # Configure train batch
        if self.algo_name in ["ppo", "ppo_serverless"]:
            trainer_config = trainer_config.training(
                train_batch_size=train_batch_size,
                sgd_minibatch_size=train_batch_size,
                num_sgd_iter=config.algos[self.algo_name]["num_sgd_iter"],
            )
        else:
            trainer_config = trainer_config.training(
                train_batch_size=train_batch_size,
                learner_queue_timeout=config.learner_queue_timeout,
                num_sgd_iter=config.algos[self.algo_name]["num_sgd_iter"],
                # learner_queue_size=config.num_rollout_workers,
                # replay_buffer_num_slots=config.num_rollout_workers,
            )

        # Configure evaluation
        trainer_config = trainer_config.evaluation(
            evaluation_interval=config.evaluation_interval,
            evaluation_num_workers=config.evaluation_num_workers,
            evaluation_duration=config.evaluation_num_workers,
        )

        # Configure async learn
        trainer_config.algo_name = self.algo_name
        trainer_config.async_learn = self.async_learn
        trainer_config.num_async_learners = config.num_async_learners
        trainer_config.num_async_return_min = config.num_async_return_min
        trainer_config.use_serverless = self.use_serverless
        trainer_config.use_weighted_grads = True
        trainer_config.use_is_truncation = True

        trainer_config.redis_host = config.redis_host
        trainer_config.redis_port = config.redis_port

        # Configure async setting of the algorithm
        trainer_config.serverless_decay = config.algos[self.algo_name]["serverless_decay"]
        trainer_config.serverless_is_ratio = config.algos[self.algo_name]["serverless_is_ratio"]
        trainer_config.serverless_lr_smooth = config.algos[self.algo_name]["serverless_lr_smooth"]

        self.trainer_config = trainer_config

    def reset_trainer(self):
        self.trainer = self.trainer_config.build()

    def reset_cost(self):
        self.cost = 0

    def get_state(
        self,
        info
    ):
        # Compute state
        if info is not None:
            round_id = info["round_id"]
            learner_loss = info["learner_loss"]
            eval_reward_mean = info["eval_reward_mean"]
            learner_kl = info["kl"]
            learner_time = info["learner_time"]
            actor_time = info["actor_time"]
        else:
            round_id = 0
            learner_loss = 0
            eval_reward_mean = 0
            learner_kl = 0
            learner_time = 0
            actor_time = 0

        state = torch.Tensor(
            [
                round_id,
                learner_loss,
                eval_reward_mean,
                learner_kl,
                learner_time,
                actor_time,
                learner_time + actor_time,
                self.budget - self.cost
            ]
        ).unsqueeze(0)

        # Compute mask
        mask = self.get_mask(state)
        
        return state, mask

    def get_mask(
        self,
        state
    ):
        return None

    def get_reward(
        self, 
        info,
        done
    ):
        return None

    def get_done(
        self,
        info
    ):
        plateau_stopper = False
        max_round_stopper = False
        budget_stopper = False

        if info is not None and info["round_id"] >= self.stop_min_round: 
            # Plateau stopper
            if len(self.stop_window) >= self.stop_grace_period:
                if info["eval_stop_cv"] <= self.stop_cv:
                    plateau_stopper = True
                else:
                    plateau_stopper = False
            else:
                plateau_stopper = False 
            
            # Maximum round stopper
            if info["round_id"] >= self.stop_max_round:
                max_round_stopper = True
            else:
                max_round_stopper = False
            
           # Budget stopper
            if self.cost >= self.budget:
                budget_stopper = True
            else:
                budget_stopper = False

        return any([plateau_stopper, max_round_stopper, budget_stopper])

    def get_info(
        self,
        round_id,
        duration,
        train_results,
    ):
        if round_id is not None and train_results is not None:
            # Learner time
            if self.algo_name in ["appo_serverless", "impala_serverless", "appo", "impala"]:
                learner_time = (train_results["info"]["timing_breakdown"]["learner_grad_time_ms"] + \
                    train_results["info"]["timing_breakdown"]["learner_load_time_ms"] + \
                    train_results["info"]["timing_breakdown"]["learner_load_wait_time_ms"] + \
                    train_results["info"]["timing_breakdown"]["learner_dequeue_time_ms"]) / 1000
            else:
                learner_time = train_results["timers"]["learn_time_ms"] / 1000
            
            # Actor time
            if self.algo_name in ["appo_serverless", "impala_serverless", "appo", "impala"]:
                actor_time = 0
            else:
                actor_time = train_results["timers"]["sample_time_ms"] / 1000

            # Eval rewards
            episode_reward = train_results['evaluation']["hist_stats"]["episode_reward"]
            if episode_reward:
                # episode_reward = utils.remove_outliers(episode_reward)
                eval_reward_max = np.max(episode_reward)
                eval_reward_mean = np.mean(episode_reward)
                eval_reward_min = np.min(episode_reward)
                eval_reward_std = np.std(episode_reward)
            else:
                eval_reward_max = 0
                eval_reward_mean = 0
                eval_reward_min = 0

            if eval_reward_mean > self.max_eval_reward:
                self.max_eval_reward = eval_reward_mean
            
            episodes_this_iter = train_results['episodes_this_iter']
            num_steps_trained_this_iter = train_results['num_steps_trained_this_iter']

            # Learner info
            if self.algo_name in ["ppo", "ppo_serverless"]:
                if DEFAULT_POLICY_ID in train_results["info"]['learner']:
                    learner_loss = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['total_loss']
                    kl = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['kl']
                    entropy = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']["entropy"]
                    async_info = train_results["info"]['learner'][DEFAULT_POLICY_ID]["async_info"]
                else:
                    learner_loss = 0
                    kl = 0
                    entropy = 0
                    async_info = []
                estimate_batch = train_results["info"]['learner']['estimate_batch']
            else:
                if DEFAULT_POLICY_ID in train_results["info"]['learner']:
                    learner_loss = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['total_loss']
                    kl = 0
                    entropy = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']["entropy"]
                    async_info = train_results["info"]['learner'][DEFAULT_POLICY_ID]["async_info"]
                else:
                    learner_loss = 0
                    kl = 0
                    entropy = 0
                    async_info = []
                estimate_batch = train_results["info"]['learner']['estimate_batch']

            # Learning stats
            episode_grad_staleness = []
            exclude_times = []
            learner_times = []
            
            for iteration in async_info:
                if "exclude_time" in iteration:
                    exclude_times.append(iteration["exclude_time"])

                episode_grad_staleness.extend(iteration["grad_staleness_list"])
                learner_times.extend(iteration["learner_times"])

            if episode_grad_staleness:
                grad_staleness_max = np.max(episode_grad_staleness)
                grad_staleness_mean = np.mean(episode_grad_staleness)
                grad_staleness_min = np.min(episode_grad_staleness)
                grad_staleness_std = np.std(episode_grad_staleness)
            else:
                grad_staleness_max = 0
                grad_staleness_mean = 0
                grad_staleness_min = 0
                grad_staleness_std = 0
            
            # Query Lambda requests
            if self.is_serverless:
                lambda_durations = self.redis_hget_lambda_duration('lambda_duration')
            else:
                lambda_durations = []

            # Stop criteria
            # self.stop_window.append(eval_reward_mean)
            # self.stop_window.append(learner_loss)
            if len(self.stop_window) > 1:
                try:
                    eval_stop_cv = utils.cv(self.stop_window)
                except Exception:
                    eval_stop_cv = float("inf")
            else:
                eval_stop_cv = float("inf")

            # Duration
            lambda_duration_max = 0
            if self.is_serverless:
                if lambda_durations:
                    lambda_duration_max = np.max(lambda_durations)

            # Timers
            timers = train_results["timers"]

            if self.use_serverless:
                duration = duration - np.sum(exclude_times)
                # if "learn_time_ms" in timers:
                #     duration = duration - timers["learn_time_ms"] / 1000

            info = {
                "round_id": round_id,
                "episodes_this_iter": episodes_this_iter,
                "duration": duration,
                "lambda_duration_max": lambda_duration_max,
                "learner_time": learner_time,
                "actor_time": actor_time,
                "eval_reward_max": eval_reward_max,
                "eval_reward_mean": eval_reward_mean,
                "eval_reward_min": eval_reward_min,
                "eval_reward_std": eval_reward_std,
                "eval_stop_cv": eval_stop_cv,
                "learner_loss": learner_loss,
                "kl": kl,
                "episode_reward": episode_reward,
                "entropy": entropy,
                "estimate_batch": estimate_batch,
                "lambda_durations": lambda_durations,
                "async_info": async_info,
                "grad_staleness_max": grad_staleness_max,
                "grad_staleness_mean": grad_staleness_mean,
                "grad_staleness_min": grad_staleness_min,
                "grad_staleness_std": grad_staleness_std,
                "timers": timers,
            }
        else:
            info = None

        return info

    def step(
        self,
        round_id,
        action,
        serverless_bound,
    ):
        duration_start_time = time.time()

        # Update action
        num_rollout_workers = action["num_rollout_workers"]
        num_envs_per_worker = action["num_envs_per_worker"]
        rollout_fragment_length = config.envs[self.env_name]['rollout_fragment_length']
        if self.algo_name in ["ppo", "ppo_serverless"]:
            train_batch_size = num_rollout_workers * num_envs_per_worker * rollout_fragment_length
        else:
            train_batch_size = num_envs_per_worker * rollout_fragment_length
        
        #
        # Update config
        #
        
        self.trainer.config.thaw()

        self.trainer.config.train_batch_size = self.trainer.get_policy().config["train_batch_size"] = train_batch_size
        self.trainer.config.num_rollout_workers = self.trainer.get_policy().config["num_rollout_workers"] = num_rollout_workers
        self.trainer.config.num_envs_per_worker = self.trainer.get_policy().config["num_envs_per_worker"] = num_envs_per_worker
        self.trainer.config.sgd_minibatch_size = self.trainer.get_policy().config["sgd_minibatch_size"] = train_batch_size

        if self.algo_name in ["ppo", "ppo_serverless"]:
            self.trainer.config.serverless_batch_min = self.trainer.get_policy().config["serverless_batch_min"] = num_rollout_workers
            self.trainer.config.estimate_batch_size = num_rollout_workers * num_envs_per_worker * rollout_fragment_length
        else:
            self.trainer.config.serverless_batch_min = self.trainer.get_policy().config["serverless_batch_min"] = config.num_rollout_workers_min
            self.trainer.config.estimate_batch_size =  num_envs_per_worker * rollout_fragment_length
        
        if self.use_serverless:
            self.trainer.config.serverless_bound = self.trainer.get_policy().config["serverless_bound"] = serverless_bound

        self.trainer.config.freeze()

        # Scale actors
        if self.is_serverless:
            payload = {
                "redis_host": config.redis_host,
                "redis_port": config.redis_port,
                "algo_name": self.lambda_algo_name,
                "env_name": self.env_name,
                "num_envs_per_worker": num_envs_per_worker,
                "rollout_fragment_length": rollout_fragment_length,
            }

            # Scale serverless actors
            self.scale_serverless_actors(
                num_rollout_workers=num_rollout_workers,
                payload=payload,
            )
        else:
            # Rebuild worker set
            self.trainer.workers._worker_manager.clear()
            self.trainer.workers.add_workers(
                num_workers=num_rollout_workers,
                validate=True
            )

            # Synchronize policy weights
            self.trainer.workers.sync_weights()

        # Train one round
        train_results = self.trainer.train()
        # print(train_results)

        duration_end_time = time.time()

        # Process train results
        info = self.get_info(
            round_id=round_id,
            duration=duration_end_time - duration_start_time,
            train_results=train_results,
        )

        # Update budget
        if self.is_serverless or self.use_serverless:
            # cost_per_round = info['lambda_duration_max'] * config.serverless_learner_per_s + sum(info['lambda_durations']) * config.serverless_actor_per_s
            cost_per_round = info['duration'] * (config.serverless_learner_per_s * config.num_async_learners + config.server_actor_per_s)
            # print("duration: {}".format(info['duration']))
            # print("lambda_durations: {}".format(info['lambda_durations']))
        else:
            cost_per_round = info['duration'] * (config.server_learner_per_s + config.server_actor_per_s)
        self.cost = self.cost + cost_per_round
        info['cost'] = self.cost

        # Get next 
        state, mask = self.get_state(info=info)
        done = self.get_done(info=info)
        reward = self.get_reward(info=info, done=done)

        return state, mask, reward, done, info

    def save(self, ckpt_path):
        save_path = None
        if self.trainer is not None:
            save_path = self.trainer.save_checkpoint(ckpt_path)
        return save_path

    def load(self, ckpt_path):
        if self.trainer is not None:
            self.trainer.load_checkpoint(ckpt_path)

    def get_policy_state(self):
        if self.trainer is not None:
            return self.trainer.get_policy().get_state()
    
    def set_policy_state(self, policy_state):
        if self.trainer is not None:
            self.trainer.get_policy().set_state(policy_state)

    def get_model_weights(self):
        return self.trainer.get_policy().get_weights()

    def invoke_serverless_actors(self, payload):
        self.lambda_client.invoke(
            FunctionName='serverless_actor',
            InvocationType='Event',
            LogType='None',
            Payload=json.dumps(payload),
        )

    def scale_serverless_actors(
        self, 
        num_rollout_workers, 
        payload
    ):
        # Set latest model weights to Redis
        self.redis_set_model_weights(self.get_model_weights())

        # Invoke actors concurrently
        jobs = []
        for _ in range(num_rollout_workers):
            p = multiprocessing.Process(
                target=self.invoke_serverless_actors,
                args=(payload,)
            )
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

    def scale_test(
        self, 
        num_rollout_workers, 
        payload
    ):
        # Set latest model weights to Redis
        self.redis_set_model_weights(self.get_model_weights())

        # Invoke actors concurrently
        jobs = []
        for _ in range(num_rollout_workers):
            p = multiprocessing.Process(
                target=self.invoke_serverless_actors,
                args=(payload,)
            )
            jobs.append(p)
            p.start()

        start_time = time.time()

        for p in jobs:
            p.join()

        invoke_end_time = time.time()

        qeury = True
        while qeury:
            if self.redis_client.hlen("lambda_duration") >= num_rollout_workers:
                qeury = False
        
        query_end_time = time.time()

        # Compute stats
        invoke_overhead = invoke_end_time - start_time
        query_overhead = query_end_time - invoke_end_time

        return invoke_overhead, query_overhead

    def prewarm_serverless_actors(self, payload):
        # Set latest model weights to Redis
        self.redis_set_model_weights(self.get_model_weights())

        # Prewarm to maximum
        jobs = []
        for _ in range(config.num_rollout_workers_max):
            p = multiprocessing.Process(
                target=self.invoke_serverless_actors,
                args=(payload,)
            )
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        # Clear Redis
        self.redis_client.flushall()

    def redis_set_model_weights(self, model_weights):
        self.redis_client.set("model_weights", pickle.dumps(model_weights))

    def redis_hget_sample_batch(self, name):
        batch_new = []
        if self.redis_client.exists(name):
            batch_all = self.redis_client.hgetall(name)
            for aws_request_id in batch_all.keys():
                if aws_request_id not in self.aws_request_id_list:
                    batch_new.append(pickle.loads(batch_all[aws_request_id]))
                    self.aws_request_id_list.append(aws_request_id)
        
        return batch_new

    def redis_hget_lambda_duration(self, name):
        lambda_duration_new = []
        if self.redis_client.exists(name):
            lambda_duration_all = self.redis_client.hgetall(name)
            for aws_request_id in lambda_duration_all.keys():
                if aws_request_id not in self.aws_request_id_list:
                    lambda_duration = float(lambda_duration_all[aws_request_id].decode())
                    lambda_duration_new.append(lambda_duration)
                    self.aws_request_id_list.append(aws_request_id)
        
        return lambda_duration_new

    def pause_learner(self):
        if self.trainer is not None:
            self.trainer.local_mixin_buffer.replay_ratio = 1.0
    
    def resume_learner(self):
        if self.trainer is not None:
            self.trainer.local_mixin_buffer.replay_ratio = self.config.replay_ratio

    def reset(self):
        self.stop_trainer()
        self.reset_trainer()
        self.reset_cost()
        
        if self.is_serverless:
            self.reset_aws_request_id_list()
        
        if self.async_learn or self.is_serverless:
            self.redis_client.flushall()
        
        self.max_eval_reward = -1e8

        info = self.get_info(
            round_id=None, 
            duration=0,
            train_results=None,
        )
        state, mask = self.get_state(info=None)

        return state, mask, info
