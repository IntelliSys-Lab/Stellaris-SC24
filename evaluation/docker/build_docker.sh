#! /bin/bash

set -ex

REPO="yhf0218"

# Build head node
cd head
docker build -t ${REPO}/stellaris-head .
cd ..

# Build worker node
cd worker 
docker build -t ${REPO}/stellaris-worker .
cd ..

# Clear dangling images
docker image prune -f
