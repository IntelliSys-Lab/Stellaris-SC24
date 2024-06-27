#! /bin/bash

# Start a ray head
/usr/local/bin/ray start --head --port=${HEAD_PORT} --include-dashboard=false --disable-usage-stats --num-cpus=16 --num-gpus=1 --memory=$((4*4*1000*1024*1024)) --object-store-memory 16000000000
tail -f /dev/null
