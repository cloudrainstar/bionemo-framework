# Access and Startup

The BioNeMo Framework is free to use and easily accessible. The preferred method of accessing the software is through the BioNeMo Docker container, which provides a seamless and hassle-free way to develop and execute code. By using the Docker container, you can bypass the complexity of handling dependencies, ensuring that you have a consistent and reproducible environment for your projects.

In this section of the documentation, we will guide you through the process of pulling the BioNeMo Docker container and setting up a local development environment. By following these steps, you will be able to quickly get started with the BioNeMo Framework and begin exploring its features and capabilities.

# Access the BioNeMo Framework

To access the BioNeMo Framework container, you will need a free NVIDIA GPU Cloud (NGC) account and an API key linked to that account.

## NGC Account and API Key Configuration

NGC is a portal of enterprise services, software, and support for artificial intelligence and high-performance computing (HPC) workloads. The BioNeMo Docker container is hosted on the NGC Container Registry. To pull and run a container from this registry, you will need to create a free NGC account and an API Key using the following steps:

1. Create a free account on [NGC](https://ngc.nvidia.com/signin) and log in.
2. At the top right, click on the **User > Setup > Generate API Key**, then click **+ Generate API Key** and **Confirm**. Copy and store your API Key in a secure location.

You can now view the BioNeMo Framework container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) or by searching the NGC Catalog for “BioNeMo Framework”. Feel free to explore the resources available to you in the catalog.

# Startup Instructions

BioNeMo is compatible with a wide variety of computing environments, including both local workstations, data centers, and Cloud Service Providers (CSPs) such as Amazon Web Service, Microsoft Azure, Google Cloud Platform, and Oracle Cloud Infrastructure, and NVIDIA’s own DGX Cloud infrastructure.

## Running the Container on a Local Machine

This section will provide instructions for running the BioNeMo Framework container on a local workstation. This process will involve the following steps:

1. Logging into the NGC Container Registry (nvcr.io)
2. Pulling the container from the registry
3. Running a Jupyter Lab instance inside the container for local development

### Pull Docker Container from NGC

Within the NGC Catalog, navigate to **BioNeMo Framework > Tags > Get Container**, and copy the image path for the latest tag.

Open a command prompt on your machine and enter the following:

```bash
docker login nvcr.io
```

This command will prompt you to enter your API key. Fill in the details as shown below. Note that you should enter the string `$oauthtoken` as your username. Replace the password (`<YOUR_API_KEY>`) with the API key that you generated in the [NGC Account and API Key Configuration](#NGC-Account-and-API-Key-Configuration) section above

```bash
Username: $oauthtoken
Password: <YOUR_API_KEY>
```

You can now pull the BioNeMo Framework container using the following command:

```bash
docker pull {{ docker_url }}:{{ docker_tag }}
```

### Run the BioNeMo Framework Container

Now that you have pulled the BioNeMo Framework container, you can run it as you would a normal Docker container. For instance, to get basic shell access you can run the following command:

```bash
docker run \
  --rm \
  -it \
  --gpus all \
  {{ docker_url }}:{{ docker_tag }} \
  /bin/bash
```

#### Running Jupyter Lab Inside the Container

First, create a local workspace directory (to be mounted to the home directory of the Docker container to persist data). You can then launch the container. We recommend running the container in a Jupyter Lab environment using the command below:

```bash
docker run --rm -d --gpus all -p 8888:8888 \
  -v <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE> \
  {{ docker_url }}:{{ docker_tag }} \
  "jupyter lab \
  	--allow-root \
	--ip=* \
	--port=8888 \
	--no-browser \
  	--NotebookApp.token='' \
  	--NotebookApp.allow_origin='*' \
  	--ContentsManager.allow_hidden=True \
  	--notebook-dir=/workspace/bionemo"
```

Let's break down this `docker run` command:

**Basic components**

* `docker run`: This is the Docker command to run a container from an image.
* `--rm`: This flag removes the container when it exits.
* `-d`: This flag runs the container in detached mode (i.e., in the background).

**Resource allocation**

* `--gpus all`: This flag allows the container to use all available GPUs on the host machine.

**Port mapping**

* `-p 8888:8888`: This flag maps port 8888 on the host machine to port 8888 inside the container. This allows access to the Jupyter Lab interface from outside the container.

**Volume mounting**

* `-v <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE>`: This flag mounts a volume from the host machine to the container. Specifically, it mounts the `<YOUR_WORKSPACE>` directory on the host machine to `/workspace/bionemo/<YOUR_WORKSPACE>` inside the container. This allows the container to access files from the host machine.

**Image and command**

* `{{ docker_url }}:{{ docker_tag }}`: This is the path to the Docker image to use.
* `"jupyter lab ..."`: This is the command to run inside the container, which is a Jupyter Lab server. The options are:
	+ `--allow-root`: Allow the Jupyter Lab server to run as the root user.
	+ `--ip=*`: Listen on all available network interfaces (i.e., allow access from outside the container).
	+ `--port=8888`: Listen on port 8888.
	+ `--no-browser`: Don't open a browser window automatically.
	+ `--NotebookApp.token=''`: Set an empty token for the Jupyter Lab server (i.e., no authentication is required).
	+ `--NotebookApp.allow_origin='*'`: Allow requests from anywhere (i.e., any origin).
	+ `--ContentsManager.allow_hidden=True`: Allow the contents manager to access hidden files and directories.
	+ `--notebook-dir=/workspace/bionemo`: Set the notebook directory to `/workspace/bionemo` inside the container.

In summary, this command runs a detached Docker container from a specified image, mapping port 8888, mounting a volume for persistent storage, and running a Jupyter Lab server with a specified configuration.
