#! /bin/bash

set -e

CONTAINER_NAME="docker-stellaris-head-1"
WORKDIR="/root/Stellaris-SC24/evaluation"

SYSTEM_NAMES=("stellaris" "rllib")
ENV_NAMES=("Hopper-v3" "Humanoid-v3" "Walker2d-v3")
ALGO_NAMES=("ppo")

# Start redis server
docker exec -w ${WORKDIR}/docker ${CONTAINER_NAME} ./redis_start.sh

# Run each one
for system_name in "${SYSTEM_NAMES[@]}"
do
    for env_name in "${ENV_NAMES[@]}"
    do
        for algo_name in "${ALGO_NAMES[@]}"
        do
            echo ""
            echo "Running ${system_name}, ${env_name}, ${algo_name} ..."
            echo ""

            docker exec -w ${WORKDIR}/experiment -t ${CONTAINER_NAME} python3 ${system_name}.py --env_name ${env_name} --algo_name ${algo_name}
        done
    done
done

# Plot results
docker exec -w ${WORKDIR}/experiment -t ${CONTAINER_NAME} python3 plot.py

# Copy figures back to host
docker cp ${CONTAINER_NAME}:${WORKDIR}/experiment/figures/ ./experiment/figures
