#!/bin/bash

set -ex

REDIS_PORT=6379

redis-server /etc/redis/redis.conf
redis-cli -p $REDIS_PORT ping 
