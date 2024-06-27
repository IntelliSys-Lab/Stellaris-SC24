#! /bin/bash

# Start a ray head
/usr/local/bin/ray start --address=${HEAD_HOSTNAME}:${HEAD_PORT} --disable-usage-stats --num-cpus=1 --num-gpus=0 --memory=$((4*1000*1024*1024)) --object-store-memory 2000000000
tail -f /dev/null
