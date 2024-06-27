FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install libraries from apt
RUN apt-get update 
RUN apt-get -y install software-properties-common ca-certificates
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update 
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update 
RUN apt-get -y install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf build-essential curl gpg gcc-9 gcc-11 g++-9 pkg-config psmisc unzip python3 python3-pip python-is-python3 wget git vim awscli net-tools redis-server jq zip

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Install bazel
RUN git clone --depth 1 https://github.com/IntelliSys-Lab/Stellaris-SC24.git /root/Stellaris-SC24
RUN cd /root/Stellaris-SC24/ci/env && ./install-bazel.sh

# Install pip dependencies
RUN cd /root/Stellaris-SC24/python && pip install -e . --verbose

# Install mujoco
RUN wget "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz" 
RUN tar -xvf mujoco210-linux-x86_64.tar.gz 
RUN mkdir /root/.mujoco
RUN mv ./mujoco210 /root/.mujoco/mujoco210 
RUN rm mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin
ENV PATH=/root/.local/bin:${PATH}

# Set head node address
ENV HEAD_HOSTNAME="head"
ENV HEAD_PORT="6380"

# Start from entrypoint
COPY entrypoint.sh /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]

