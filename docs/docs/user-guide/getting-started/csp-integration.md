# Running the BioNeMo Container with Major Cloud Service Providers

Using the BioNeMo Framework container on a cloud service provider (CSP) allows researchers to access scalable computing
resources, eliminates infrastructure management, and enables seamless collaboration and deployment, ultimately
accelerating AI model development in the life sciences. Below, we outline several strategies and supported CSPs with
which you can get started using the BioNeMo Framework.

## Running the Container on NVIDIA DGX Cloud

For DGX Cloud users, NVIDIA Base Command Platform (BCP) includes a central user interface with managed compute
resources. BCP can be used to manage datasets, workspaces, jobs, and users within an organization and team. This setup
creates a convenient hub for monitoring job execution, viewing metrics and logs, and monitoring resource utilization.
More information can be found on the [BCP website](https://docs.nvidia.com/base-command-platform/index.html).

### NGC CLI Configuration

NVIDIA NGC Command Line Interface (CLI) is a command-line tool for managing Docker containers in NGC. You can download
it on your local machine using the instructions [on the CLI website](https://org.ngc.nvidia.com/setup/installers/cli).

Once you have installed the CLI, run `ngc config set` at the command line to setup your NGC credentials:

* **API key**: Enter your API Key
* **CLI output**: Accept the default (ascii format) by pressing `Enter`
* **org**: Choose your preferred organization from the list
* **team**: Choose the team to which you have been assigned
* **ace** (optional): Choose an ACE, if applicable, otherwise press `Enter` to continue

Note that the **org** and **team** are only relevant when pulling private containers/datasets from NGC created by you or
your team. For BioNeMo Framework, you can use the default value.


!!! note
    The NGC documentation also discusses how to mount your own
    [datasets](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-datasets) and
    [workspaces](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#managing-workspaces).

### Running the BioNeMo Framework Container on DGX Cloud

On your local machine, run the following command to launch your job, replacing the relevant fields with your settings:

```bash
ngc batch run \
    --name <YOUR_JOB_NAME> \
    --team <YOUR_TEAM> \
    --ace <YOUR_ACE> \
    --instance dgxa100.80g.1.norm \
    --image {{ docker_url }}:{{ docker_tag }} \
    --port 8888 \
    --workspace <YOUR_WORKSPACE>:/workspace/bionemo/<YOUR_WORKSPACE>:RW \
    --datasetid <YOUR_DATASET> \
    --result /result \
    --total-runtime 1D \
    --order 1 \
    --label <YOUR_LABEL> \
    --commandline "jupyter lab \
        --allow-root \
        --ip=* \
        --port=8888 \
        --allow-root \
        --no-browser \
        --NotebookApp.token='' \
        --NotebookApp.allow_origin='*' \
        --ContentsManager.allow_hidden=True \
        --notebook-dir=/workspace/bionemo & sleep infinity"
```

Explanation:

* `--name`: Name of your job
* `--team`: Team that you are assigned in NGC org
* `--ace` (optional): ACE that you are assigned
* `--instance`: GPU instance type for the job (for example, `dgxa100.80g.1.norm` for single-GPU A100 instance)
* `--image`: BioNeMo Framework container image
* `--port`: Port number to access JupyterLab
* `--workspace` (optional): Mount NGC workspace to container with read/write access to persist data
* `--datasetid` (optional): Mount dataset to container
* `--result`: Directory to store job results
* `--order`: Order of the job
* `--label`: Job label, allowing quick filtering on NGC dashboard
* `--commandline`: Command to run inside the container, in this case, starting JupyterLab and keeping it running with
    `sleep infinity`

To launch your Jupyter notebook in the browser, click on your job in the [NGC Web UI](https://bc.ngc.nvidia.com/jobs)
and then click the URL under the Service Mapped Ports. You may also set up a Remote Tunnel to access a running job to
execute and edit your code using VS Code locally or via the browser, as discussed
this page in the [BCP documentation](https://docs.nvidia.com/base-command-platform/user-guide/latest/index.html#setting-up-and-accessing-visual-studio-code-via-remote-tunnel).

## Running on Any Major CSP with the NVIDIA GPU-Optimized VMI

The BioNeMo Framework container is supported on cloud-based GPU instances through the
**NVIDIA GPU-Optimized Virtual Machine Image (VMI)**, available for
[AWS](https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq#pdp-pricing),
[GCP](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi),
[Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.ngc_azure_17_11?tab=overview), and
[OCI](https://cloudmarketplace.oracle.com/marketplace/en_US/listing/165104541).
NVIDIA VMIs are built on Ubuntu and provide a standardized operating system environment across cloud infrastructure for
running NVIDIA GPU-accelerated software. These images are pre-configured with software dependencies such as NVIDIA GPU
drivers, Docker, and the NVIDIA Container Toolkit. More details about NVIDIA VMIs can be found in the
[NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nvidia_vmi).

The general steps for launching the BioNeMo Framework container using a CSP are:

1. Launch a GPU instance running the NVIDIA GPU-Optimized VMI on your preferred CSP. Follow the instructions for
    launching a GPU-equipped instance provided by your CSP.
2. Connect to the running instance using SSH and run the BioNeMo Framework container exactly as outlined in the
    [Running the Container on a Local Machine](./access-startup.md#running-the-container-on-a-local-machine) section on
    the Access and Startup page.

### Integration with Managed Cloud Services

BioNeMo is also compatible with various managed services from these cloud providers. Check out blogs about BioNeMo on
[SageMaker](https://aws.amazon.com/blogs/industries/find-the-next-blockbuster-with-nvidia-bionemo-framework-on-amazon-sagemaker/)
(example code [repository](https://github.com/aws-samples/amazon-sagemaker-with-nvidia-bionemo)),
[ParallelCluster](https://aws.amazon.com/blogs/hpc/protein-language-model-training-with-nvidia-bionemo-framework-on-aws-parallelcluster/)
(example code [repository](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/14.bionemo)),
and [EKS](https://aws.amazon.com/blogs/hpc/accelerate-drug-discovery-with-nvidia-bionemo-framework-on-amazon-eks/)
(example code [repository](https://github.com/awslabs/data-on-eks/tree/main/ai-ml/bionemo)).
