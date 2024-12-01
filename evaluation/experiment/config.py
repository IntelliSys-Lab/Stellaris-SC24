import socket


# Paths
log_path = "./logs"
figure_path = "./figures"
ckpt_path = "./ckpt"

# Ray cluster
num_rollout_workers = 16
num_envs_per_worker = 1
num_cpus_for_local_worker = 8
num_gpus_for_local_worker = 1
num_cpus_per_worker = 1
num_gpus_per_worker = 0
evaluation_num_workers = 10
min_time_s_per_iteration = None
_enable_new_api_stack = False
learner_queue_timeout = 114514

# Redis
# redis_host = socket.gethostbyname(socket.gethostname())
redis_host = socket.gethostbyname("localhost")
redis_port = 6379

# Asynchronous learning
async_learn = True
num_async_learners = 4
num_async_return_min = 1

# Training
max_exp = 1
framework = "torch"
envs = {
    "Hopper-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 600,
        "budget": float("inf"),
        "rollout_fragment_length": 256,
    },
    "Humanoid-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 1000,
        "budget": float("inf"),
        "rollout_fragment_length": 256,
    },
    "Walker2d-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 1000,
        "budget": float("inf"),
        "rollout_fragment_length": 256,
    },
}
algos = {
    "ppo": {
        "num_sgd_iter": 10,
        "serverless_decay": 0.9,
        "serverless_is_ratio": 1.0,
        "serverless_lr_smooth": 3,
    },
}

# Stop criteria
stop_num_results = 5
stop_cv = 0.0001
stop_grace_period = stop_num_results
stop_min_round = 1
stop_max_round = 20

# PPO to be trained
clip_param = 0.2
# num_sgd_iter = 32
vf_clip_param = 100
entropy_coeff = 0.01
kl_coeff = 0

# Evaluate
evaluation_interval = 1

# Pricing units
serverless_learner_per_s = (3.0600 + 17.92 / 30 / 24) / 60 / 60 / num_async_learners # vm + ip + disk
serverless_actor_per_s = 0.0000000167 * 1000  # mem + network
server_learner_per_s = (3.0600 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
server_actor_per_s = (2.04 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
