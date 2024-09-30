# Running the BioNeMo Container with Major Cloud Service Providers

## Launch Instance Through NVIDIA VMI

The BioNeMo Framework container is supported on cloud-based GPU instances through the **NVIDIA GPU-Optimized Virtual Machine Image (VMI)**, available for [AWS](https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq#pdp-pricing), [GCP](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi), [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=overview), and [OCI](https://cloudmarketplace.oracle.com/marketplace/en_US/listing/165104541). NVIDIA VMIs are built on Ubuntu and provide a standardized operating system environment across cloud infrastructure for running NVIDIA GPU-accelerated software. These images are pre-configured with software dependencies such as NVIDIA GPU drivers, Docker, and the NVIDIA Container Toolkit. More details about NVIDIA VMIs can be found in the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nvidia_vmi).

The general steps for launching the BioNeMo Framework container in :

1. Launch a GPU instance running the NVIDIA GPU-Optimized VMI on your preferred CSP. Follow the instructions for launching a GPU-equipped instance provided by your CSP.
2. Connect to the running instance using SSH and run the BioNeMo Framework container exactly as outlined in the [Running the Container on a Local Machine](#running-the-container-on-a-local-machine) section above.

### Integration with Cloud Services

BioNeMo is also compatible with various cloud services. Check out blogs about BioNeMo on [SageMaker](https://aws.amazon.com/blogs/industries/find-the-next-blockbuster-with-nvidia-bionemo-framework-on-amazon-sagemaker/) (example code [repository](https://github.com/aws-samples/amazon-sagemaker-with-nvidia-bionemo)), [ParallelCluster](https://aws.amazon.com/blogs/hpc/protein-language-model-training-with-nvidia-bionemo-framework-on-aws-parallelcluster/) (example code [repository](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/14.bionemo)), and [EKS](https://aws.amazon.com/blogs/hpc/accelerate-drug-discovery-with-nvidia-bionemo-framework-on-amazon-eks/) (example code [repository](https://github.com/awslabs/data-on-eks/tree/main/ai-ml/bionemo)).

## Running the Container on DGX Cloud

For DGX Cloud users, NVIDIA Base Command Platform (BCP) includes a central user interface with managed compute resources. It can be used to manage datasets, workspaces, jobs, and users within an organization and team. This creates a convenient hub for monitoring job execution, viewing metrics and logs, and monitoring resource utilization. NVIDIA DGX Cloud is powered by Base Command Platform. More information can be found on the [BCP website](https://docs.nvidia.com/base-command-platform/index.html).

### NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC. You can download it on your local machine as per the instructions [here](https://org.ngc.nvidia.com/setup/installers/cli).

Once installed, run `ngc config set` to establish NGC credentials:

* **API key**: Enter your API Key
* **CLI output**: Accept the default (ascii format) by pressing `Enter`
* **org**: Choose from the list which org you have access to
* **team**: Choose the team you are assigned to
* **ace**: Choose an ACE, otherwise press `Enter` to continue

Note that the **org** and **team** are only relevant when pulling private containers/datasets from NGC created by you or your team. For BioNeMo Framework, use the default value.

You can learn more about NGC CLI installation [here](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#installing-ngc-cli). Note that the NGC documentation also discusses how to mount your own [datasets](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-datasets) and [workspaces](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-workspaces).

### Running the BioNeMo Framework Container

On your local machine, run the following command to launch your job, ensuring to replace the relevant fields with your settings:

```bash
ngc batch run \
    --name <YOUR_JOB_NAME> \
    --team <YOUR_TEAM> \
    --ace <YOUR_ACE> \
    --instance dgxa100.80g.1.norm \
    --image <IMAGE_PATH> \
    --port 8888 \
    --workspace <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE>:RW \
    --datasetid <YOUR_DATASET> \
    --result /result \
    --total-runtime 1D \
    --order 1 \
    --label <YOUR_LABEL> \
    --commandline "jupyter lab --allow-root --ip=* --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --ContentsManager.allow_hidden=True --notebook-dir=/workspace/bionemo & sleep infinity"
```

Explanation:

* `--name`: Name of your job
* `--team`: Team that you are assigned in NGC org
* `--ace`: ACE that you are assigned
* `--instance`: GPU instance type for the job (e.g. `dgxa100.80g.1.norm` for single-GPU A100 instance)
* `--image`: BioNeMo Framework container image
* `--port`: Port number to access JupyterLab
* `--workspace`: Optional (Mount NGC workspace to container with read/write access to persist data)
* `--datasetid`: Optional (Mount dataset to container)
* `--result`: Directory to store job results
* `--order`: Order of the job
* `--label`: Job label, allowing quick filtering on NGC dashboard
* `--commandline`: Command to run inside the container, in this case, starting JupyterLab and keeping it running with `sleep infinity`

To launch your Jupyter notebook in the browser, click on your job in the NGC Web UI and then click the URL under the Service Mapped Ports. You may also set up a Remote Tunnel to access a running job to execute and edit your code using VS Code locally or via the browser, as discussed [here](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#setting-up-and-accessing-visual-studio-code-via-remote-tunnel).
