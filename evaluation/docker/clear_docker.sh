#! /bin/bash

set -ex

docker rm -v -f $(docker ps -qa)
docker image prune -f
