# Initialization Guide

!!! note

    Prior to beginning this section, you must confirm that your computing platform meets or exceeds the prerequisites
    outlined in the [Hardware and Software Prerequisites](./pre-reqs.md) page and that you have already pulled and
    verified that you can run the BioNeMo container as outlined in the [Access and Startup](./access-startup.md) page.

At this point, you have successfully launched and entered the Docker container. This section will guide you through
common `docker run` options to start the container, the layout of the container, and downloading pre-trained model
weights.

## Setting Up Your Host Machine Environment

We recommend using a `.env` file in your local workspace to define several useful environment variables. Specifically,
the following variables are useful to include in your `.env` file:

```txt
# Local Cache Directories
LOCAL_RESULTS_PATH
DOCKER_RESULTS_PATH
LOCAL_DATA_PATH
DOCKER_DATA_PATH
LOCAL_MODELS_PATH
DOCKER_MODELS_PATH

# Desired Jupyter Port
JUPYTER_PORT

# NGC Configuration Settings
NGC_CLI_API_KEY
NGC_CLI_ORG
NGC_CLI_TEAM
NGC_CLI_FORMAT_TYPE

# Weights and Biases API Key
WANDB_API_KEY
```

Refer to the list below for an explanation of each of these variables:

- `LOCAL_RESULTS_PATH` and `DOCKER_RESULTS_PATH`: paths for storing results, with `LOCAL` referring to the local machine
    and `DOCKER` referring to a Docker container.
- `LOCAL_DATA_PATH` and `DOCKER_DATA_PATH`: paths for storing data, again with `LOCAL` and `DOCKER` distinctions.
- `LOCAL_MODELS_PATH` and `DOCKER_MODELS_PATH`: paths for storing machine learning models, with the same local and.
    Docker differences.
- `JUPYTER_PORT`: the port number for a Jupyter Lab server.
-  `NGC_CLI_API_KEY`, `NGC_CLI_ORG`, `NGC_CLI_TEAM`, and `NGC_CLI_FORMAT_TYPE`: API key, organization, team, and format
    type for the NVIDIA GPU Cloud (NGC) command-line interface (CLI).
- `WANDB_API_KEY`: an API key for Weights and Biases (W&B), a platform for machine learning experiment tracking and
    visualization.

For each of these variables, you can define them using `=`. For example, you can set the NGC API key using
`NGC_CLI_API_KEY=<you API key here>`. You can then define these variables in your shell using:

```bash
source .env
```

Running this command will make these variables available for use in the `docker run` command examples shown below.

!!! note "Weights and Biases Setup (Optional)"

    [Weights and Biases](https://wandb.ai/) (W&B) is a machine learning operations platform that provides tools and
    services to help machine learning practitioners and teams build, train, and deploy models more efficiently. BioNeMo
    is built to work with W&B and requires only simple setup steps to start tracking your experiments. To set up W&B
    inside your container, follow the steps below:

    1. Sign up for an account at [Weights and Biases](https://wandb.ai/).
    2. Setup your [API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) with W&B.
    3. Set the `WANDB_API_KEY` variable in your `.env` in the same way as you set the previous environment variable
        above.
    4. Set the environment variable inside your container using the `-e` option, as shown in the next section.

## Common `docker run` Options



### Mounting Volumes with the `-v` Option

The `-v`  allows you to mount a host machine's directory as a volume inside the
container. This enables data persistence even after the container is deleted or restarted. In the context of machine
learning workflows, leveraging the `-v` option is essential for maintaining a local cache of datasets, model weights, and
results on the host machine such that they can persist after the container terminates and be reused across container
runs.

**Syntax:**

```
docker run -v <host_directory>:<container_directory> <image_name>
```
**Example:**

```
docker run -v /path/to/local/cache:/workspace/bionemo2/cache \
    {{ docker_url }}:{{ docker_tag }}
```

In this example, the `/path/to/local/cache` directory on the host machine is mounted as a volume at
`/workspace/bionemo2/cache` inside the container.

### Setting Environment Variables with the `-e` Option

The `-e` option allows you to set environment variables inside the container. You can use this option to define
variables that will be available to the application running inside the container.

**Example:**

```bash
docker run -e MY_VAR=value -e ANOTHER_VAR=another_value \
    {{ docker_url }}:{{ docker_tag }}
```

- `-e MY_VAR=value` sets the `MY_VAR` environment variable to `value` inside the container.
- `-e ANOTHER_VAR=another_value` sets the `ANOTHER_VAR` environment variable to `another_value` inside the container.

You can set multiple environment variables by repeating the `-e` option. The values of these variables will be available
to the application running inside the container, allowing you to customize its behavior.

Note that you can also use shell variables and command substitutions to set environment variables dynamically. For
example:

```bash
MY_EXTERNAL_VAR=external_value
docker run -e MY_INTERNAL_VAR=$MY_EXTERNAL_VAR \
    {{ docker_url }}:{{ docker_tag }}
```

In this example, the `MY_INTERNAL_VAR` environment variable inside the container will be set to the value of the
`MY_EXTERNAL_VAR` shell variable on the host machine.

### Setting User and Group IDs with the `-u` Option

The `-u` option sets the user and group IDs to use for the container process. By matching the IDs of the user on the
host machine, the user inside the container will have identical permissions for reading and writing files in the mounted
volumes as the user that ran the command. You can use command substitutions to automatically retrieve your user and
group IDs.

**Example:**

```bash
docker run -u $(id -u):$(id -g) \
    {{ docker_url }}:{{ docker_tag }}
```

- `$(id -u)` is a command substitution that executes the id -u command and captures its output. `id -u` prints the
    effective user ID of the current user.
- `$(id -g)` is another command substitution that executes the `id -g` command and captures its output. `id -g` prints
    the effective group ID of the current user.

## Starting the BioNeMo Container for Common Workflows

Below we describe some common BioNeMo workflows, including how to setup and run the container in each case.

### Running a Model Training Script Inside the Container

```bash
docker run --rm -d --gpus all -p 8888:8888 -u $(id -u):$(id -g) \
  -v <YOUR_WORKSPACE>:/workspace/bionemo3/<YOUR_WORKSPACE> \
  {{ docker_url }}:{{ docker_tag }} \
  "jupyter lab \
  	--allow-root \
	--ip=* \
	--port=8888 \
	--no-browser \
  	--NotebookApp.token='' \
  	--NotebookApp.allow_origin='*' \
  	--ContentsManager.allow_hidden=True \
  	--notebook-dir=/workspace/bionemo2/<YOUR_WORKSPACE>"
```

### Running Jupyter Lab Inside the Container

First, create a local workspace directory (to be mounted to the home directory of the Docker container to persist data).
You can then launch the container. We recommend running the container in a Jupyter Lab environment using the command
below:

```bash
docker run --rm -d --gpus all -p 8888:8888 -u $(id -u):$(id -g) \
  -v <YOUR_WORKSPACE>:/workspace/bionemo3/<YOUR_WORKSPACE> \
  {{ docker_url }}:{{ docker_tag }} \
  "jupyter lab \
  	--allow-root \
	--ip=* \
	--port=8888 \
	--no-browser \
  	--NotebookApp.token='' \
  	--NotebookApp.allow_origin='*' \
  	--ContentsManager.allow_hidden=True \
  	--notebook-dir=/workspace/bionemo2/<YOUR_WORKSPACE>"
```

Let's break down this `docker run` command:

**Basic Components**

* `docker run`: This is the Docker command to run a container from an image.
* `--rm`: This flag removes the container when it exits.
* `-d`: This flag runs the container in detached mode (i.e., in the background).
* `-u $(id -u):$(id -g)`: This option sets the user and group IDs to match those of the user running on the host machine.

**Resource Allocation**

* `--gpus all`: This option allows the container to use all available GPUs on the host machine.

**Port Mapping**

* `-p 8888:8888`: This option maps port 8888 on the host machine to port 8888 inside the container. This allows access to
the Jupyter Lab interface from outside the container.

**Volume Mounting**

* `-v <YOUR_WORKSPACE>:/workspace/bionemo2/<YOUR_WORKSPACE>`: This option mounts a volume from the host machine to the
container. Specifically, it mounts the `<YOUR_WORKSPACE>` directory on the host machine to
`/workspace/bionemo2/<YOUR_WORKSPACE>` inside the container. This configuration allows the container to access files
from the host machine.

**Image and Command**

* `{{ docker_url }}:{{ docker_tag }}`: This is the path to the Docker image to use.
* `"jupyter lab ..."`: This is the command to run inside the container, which starts a Jupyter Lab server. The options are:
	+ `--allow-root`: Allow the Jupyter Lab server to run as the root user.
	+ `--ip=*`: Listen on all available network interfaces, which allows access from outside the container.
	+ `--port=8888`: Listen on port 8888.
	+ `--no-browser`: Do not open a browser window automatically.
	+ `--NotebookApp.token=''`: Set an empty token for the Jupyter Lab server (no authentication is required).
	+ `--NotebookApp.allow_origin='*'`: Allow requests from any origin.
	+ `--ContentsManager.allow_hidden=True`: Allow the contents manager to access hidden files and directories.
	+ `--notebook-dir=/workspace/bionemo2/<YOUR_WORKSPACE>`: Set the notebook directory to
        `/workspace/bionemo/<YOUR_WORKSPACE>` inside the container.

In summary, this command runs a detached Docker container from a specified image, mapping port 8888, mounting a volume
for persistent storage, and running a Jupyter Lab server with a specified configuration.
