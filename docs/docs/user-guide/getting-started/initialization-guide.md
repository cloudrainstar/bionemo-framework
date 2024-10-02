# Initialization Guide

!!! note

    Prior to beginning this section, you must confirm that your computing platform meets or exceeds the prerequisites
    outlined in the [Hardware and Software Prerequisites](./pre-reqs.md) page.

At this point, you have successfully launched and entered the Docker container.This section will guide you through the
container, initial steps such as configuration and downloading pre-trained model weights, and where to find tutorials.

## NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC.
If NGC is not already installed in the container, download it as per the instructions on the
[CLI Installation Page](https://org.ngc.nvidia.com/setup/installers/cli) (note that within the container,
the AMD64 Linux version should be installed.)

Once installed, run `ngc config set` to establish NGC credentials within the container.

## First-Time Setup

First, invoke the following launch script. The first time, it will create a .env file and exit:

```bash
./launch.sh
```

Next, edit the .env file with the correct NGC parameters for your organization and team:

```bash
    NGC_CLI_API_KEY=<YOUR_API_KEY>
    NGC_CLI_ORG=<YOUR_ORG>
    NGC_CLI_TEAM=<YOUR_TEAM>
```

## Download Model Weights

You may now download all pre-trained model checkpoints from NGC through the following command:

```bash
./launch.sh download
```

This command will download all models to the `workspace/bionemo/models` directory. Optionally, you may persist the
models by copying them to your mounted workspace, so that they need not be re-downloaded each time.

## Directory Structure

Note that `workspace/bionemo` is the home directory for the container. Below are a few key components:

* `bionemo`: Contains the core BioNeMo package, which includes base classes for BioNeMo data modules, tokenizers,
models, etc.
* `examples`: Contains example scripts, datasets, YAML files, and notebooks
* `models`: Contains all pre-trained models checkpoints in `.nemo` format.

## Weights and Biases Setup (Optional)

Training progress and charts of the models can be visualized through
[Weights and Biases](https://docs.wandb.ai/guides/track/public-api-guide). Setup your
[API Key](https://docs.wandb.ai/guides/track/public-api-guide#authentication) to enable logging.
