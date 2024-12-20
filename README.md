<!--
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
-->

# Stellaris-SC24

[![DOI](https://zenodo.org/badge/820219752.svg)](https://zenodo.org/doi/10.5281/zenodo.12589953)

----

This repo contains a demo implementation of our SC 2024 paper, [Stellaris: Staleness-Aware Distributed Reinforcement Learning with Serverless Computing](https://dl.acm.org/doi/10.1109/SC41406.2024.00045).

> Deep reinforcement learning (DRL) has achieved remarkable success in diverse areas, including gaming AI, scientific simulations, and large-scale (HPC) system scheduling. DRL training, which involves a trial-and-error process, demands considerable time and computational resources. To overcome this challenge, distributed DRL algorithms and frameworks have been developed to expedite training by leveraging large-scale resources. However, existing distributed DRL solutions rely on synchronous learning with serverful infrastructures, suffering from low training efficiency and overwhelming training costs.
This paper proposes Stellaris, the first to introduce a generic asynchronous learning paradigm for distributed DRL training with serverless computing. We devise an importance sampling truncation technique to stabilize DRL training and develop a staleness-aware gradient aggregation method tailored to the dynamic staleness in asynchronous serverless DRL training. Experiments on AWS EC2 regular testbeds and HPC clusters show that Stellaris outperforms existing state-of-the-art DRL baselines by achieving 2.2× higher rewards (i.e., training quality) and reducing 41% training costs.

----

Stellaris is built on top of [Ray RLlib](https://github.com/apache/openwhisk). We describe how to build and deploy Stellaris from scratch for this demo. We also provide Docker container images for fast reproducing the demo experiment.

## General Hardware Prerequisite
- Operating systems and versions: Ubuntu 22.04
- Resource requirement
  - CPU: >= 36 cores
  - Memory: >= 100 GB
  - Disk: >= 50 GB
  - Network: no requirement since it's a single-node demo

## Chameleon Cloud

### Prerequisite

- [Chameleon Cloud UC access](https://chi.uc.chameleoncloud.org/): Instance node type must be **gpu_rtx_6000**
- [Chameleon Cloud image](https://chi.uc.chameleoncloud.org/project/images): Image must be **CC-Ubuntu22.04**

### Instance Setup

1. Create a lease to reserve hosts: **Reservations** -> **Leases** -> **Hosts** -> **Reserve Hosts** -> **Resource Properties** -> **node_type** -> **gpu_rtx_6000**.
2. Launch a **GPU RTX 6000** instance using the image **CC-Ubuntu22.04**. The image can be found by searching the image name: **Project** -> **Compute** -> **Images** -> **Search** -> **Launch**. Create a key pair if neccessary.
2. Assign a floating IP to your instance for login: **Project** -> **Network** -> **Floating IPs** -> **Allocate IP To Project**. 

We refer readers to [Chameleon cloud documents](https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html) for instance creation and login details.

## Deployment Instructions

1. Download the GitHub repo.
```
git clone https://github.com/IntelliSys-Lab/Stellaris-SC24
```
2. Go to [`Stellaris-SC24/evaluation`](https://github.com/IntelliSys-Lab/Stellaris-SC24/tree/master/evaluation).
```
cd Stellaris-SC24/evaluation
```
3. Install Docker container library.
```
./install_docker.sh
```
4. Install NVIDIA CUDA driver.
```
./install_nvidia.sh
```
5. Pull the Docker images directly from Docker Hub. Note that there is a [rate limit](https://docs.docker.com/docker-hub/download-rate-limit/) for image downloading per 6 hours. 
```
cd docker && ./pull_docker.sh
```
6. Start the container cluster using Docker Compose.
```
docker compose up -d
```
7. Run Stellaris demo. The demo experiment may take up to 20 minutes to complete.
```
cd ../ && ./run_experiment.sh
```

## Build Docker Images

Alternatively, we also provide scripts that build Docker images locally, but this can take a significant amount of time if built from scratch.
```
cd docker && ./build_docker.sh
```

## Results and Figures

After `run_experiment.sh` completes, you should be able to check the results and figures of training efficiency and training cost under `Stellaris-SC24/evaluation/experiment/figures`.

## Experimental Settings and Workloads

The experiment settings can be found in [`Stellaris-SC24/evaluation/experiment/config.py`](https://github.com/IntelliSys-Lab/Stellaris-SC24/tree/master/evaluation/experiment/config.py). We compare Stellaris with [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) by running the famous [PPO](https://arxiv.org/abs/1707.06347) algorithm in this demo. However, due to time and hardware limits, we only use three [Mujoco](https://github.com/openai/mujoco-py) environments.
