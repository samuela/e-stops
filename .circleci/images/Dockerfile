FROM circleci/python:3.7

# For some reason the working dir is / which the default user doesn't have
# access to so we cd into /home/circleci.
RUN cd && \
    wget https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    rm mujoco200_linux.zip && \
    mkdir ~/.mujoco && \
    mv mujoco200_linux ~/.mujoco/mujoco200

ENV LD_LIBRARY_PATH="/home/circleci/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"

RUN sudo apt-get update && \
    sudo apt-get install libosmesa6-dev libglew-dev patchelf libglfw3-dev
