# research

[![CircleCI](https://circleci.com/gh/samuela/research.svg?style=svg&circle-token=8cdcd12f566758fdc366a319545cd8343551eb0f)](https://circleci.com/gh/samuela/research)

ALL THE CODEZ

## New machine setup

On AWS:

1. Create a new VM with Ubuntu 18.04.
2. Assign it a new Elastic IP.

Locally:

1. Update `~/.ssh/config` with a new entry:

```
Host <name>
 HostName <ip address/url>
 User ubuntu
 IdentityFile ~/.ssh/aws-macbookpro.pem    # or wherever.
```

On the machine:

First,

```bash
sudo apt update
sudo apt upgrade
sudo reboot
```

1. Set the hostname.
2. Install nuvemfs.
3. Install mujoco.
4. Install linuxbrew and pipenv.

### Set hostname

On AWS Ubuntu 18.04,

```bash
user$ sudo su
root$ hostnamectl set-hostname <whatever>
```

### Install nuvemfs

```bash
sudo apt install -y cifs-utils
wget https://nuvemfscliassets.blob.core.windows.net/nuvemfs-cli-assets/stable/nuvemfs-cli-x86_64-unknown-linux-musl
chmod +x nuvemfs-cli-x86_64-unknown-linux-musl
echo "alias nuvemfs=\"~/nuvemfs-cli-x86_64-unknown-linux-musl\"" >> ~/.profile
source ~/.profile
```

### Mujoco/Ubuntu setup

1. Download and install Mujoco.

```bash
sudo apt install -y unzip clang
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin" >> ~/.profile
source ~/.profile
```

2. Install dependencies

```bash
# libosmesa6-dev: Fixes `fatal error: GL/osmesa.h: No such file or directory`
# libglew-dev: Fixes `/usr/bin/ld: cannot find -lGL`
# ffmpeg: Necessary for mujoco videos.
sudo apt install -y libosmesa6-dev libglew-dev ffmpeg

# These seem to be only necessary on circleci/python:
# patchelf: Fixes `No such file or directory: 'patchelf'`.
# libglfw3-dev: Fixes `ImportError: Failed to load GLFW3 shared library.`.
sudo apt install -y patchelf libglfw3-dev
```

Either clang will need to be set it as the default `cc` alternative (`sudo update-alternatives --config cc`) or you'll need to use gcc version 8. If you follow these instructions exactly (without ever installing `build-essentials`) then it should work no problemo.

Logging in/out to fix `$PATH` may also be necessary.

See

- https://github.com/openai/mujoco-py/issues/455
- https://github.com/openai/mujoco-py/issues/394
- https://github.com/ethz-asl/reinmav-gym/issues/35

3. Put the license key at `~/.mujoco/mjkey.txt`.

```bash
cp ~/nu/skainswo/mjkey.txt ~/.mujoco/mjkey.txt
```

### Install linuxbrew and pipenv

See https://docs.brew.sh/Homebrew-on-Linux.

```bash
# See https://stackoverflow.com/questions/24426424/unattended-no-prompt-homebrew-installation-using-expect.
echo | sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> ~/.profile
source ~/.profile
brew install pipenv
```

### CUDA/cuDNN setup

The `nvidia-driver-430` and `nvidia-cuda-toolkit` on Ubuntu 18.04 install CUDA 9.1 which is not supported by JAX at the moment.

1. Remove any current installation.

```bash
sudo apt-get purge *cuda*
sudo apt-get purge *nvidia*
sudo apt-get purge *cudnn*
```

and then follow the runfile uninstall steps (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-uninstallation).

2. Make sure that gcc is current cc alternative:

```bash
sudo update-alternatives --config cc
cc --version
```

(This was necessary for CUDA 10.1. May not be necessary for 10.0.)

3. Follow the installation instructions [here](https://developer.nvidia.com/cuda-downloads) for the "runfile (local)" version. Install version 10.0 since TF and pytorch do not yet support 10.1.

4. Add

```bash
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

to `~/.profile`.

5. Download the "cuDNN Library for Linux" (https://developer.nvidia.com/rdp/cudnn-download), not the deb version. You'll need to be logged in order for the downloads to work. Using wget/curl isn't sufficient. Easiest to download them locally and then scp them to the remote machine.

6. Install cuDNN (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar) but note that the CUDA installation directory is `/usr/local/cuda-10.0` not `/usr/local/cuda`.

7. Reboot.

8. Follow the pip instructions here (https://github.com/google/jax#pip-installation) in a `pipenv shell` to install the new GPU versions of `jax`/`jaxlib`.

See

- https://developer.nvidia.com/cuda-zone
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#faq2
- https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible
- https://discuss.pytorch.org/t/when-pytorch-supports-cuda-10-1/38852

Note that the deb installation does not seem to support multiple CUDA installations living in harmony. This may become problematic as some packages like pytorch do not yet support CUDA 10.1.

With CUDA 10.0, JAX may require the `xla_gpu_cuda_data_dir` XLA flag to be set as well:

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.0/
```

### Expand EBS volume

No downtime is necessary.

1. Change the volume in the console.
2. Then follow https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html. Use `df -T` to get the filesystem type.

## Mujoco lockfile issues

Sometimes the mujoco lockfile gets screwed up and in that case it's necessary to delete it. If jobs are just hanging forever without starting try deleting the lockfile:

```bash
rm $(pipenv --venv)/lib/python3.7/site-packages/mujoco_py/generated/mujocopy-buildlock.lock
```

See https://github.com/openai/mujoco-py/issues/424.
