FROM nvcr.io/nvidia/tensorrt:19.05-py3
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && apt update && apt install sudo
RUN useradd -ms /bin/bash rt-ros-docker
RUN adduser rt-ros-docker sudo
RUN echo "rt-ros-docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER rt-ros-docker
WORKDIR /home/rt-ros-docker
RUN sudo apt update && sudo apt install -y lsb-release && \
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 && \
    sudo apt update && \
    sudo apt install -y ros-kinetic-desktop-full && \
    sudo rosdep init && \
    rosdep update && \
    sudo apt install -y python-rosinstall \
         python-rosinstall-generator \ 
         python-wstool \
         build-essential && \
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc && \
    source ~/.bashrc && \
    sudo apt-get install -y libqt4-dev 
         qt4-dev-tools \ 
         libglew-dev \ 
         glew-utils \ 
         libgstreamer1.0-dev \ 
         libgstreamer-plugins-base1.0-dev \ 
         libglib2.0-dev

