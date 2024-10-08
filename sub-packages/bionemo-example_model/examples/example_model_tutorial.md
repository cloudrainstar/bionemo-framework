This tutorial demonstrates the creation of a model in BioNeMo.
`Megatron` / `NeMo` modules and datasets are special derivatives of PyTorch modules and datasets that extend and accelerate the distributed training and inference capabilities of PyTorch.

Some distinctions of Megatron / NeMo are:

- `torch.nn.Module`/`LightningModule` changes into `MegatronModule`.
- Loss functions should extend the `MegatronLossReduction` module and implement a `reduce` method for aggregating loss across multiple micro-batches.
- Megatron configuration classes (e.g. `megatron.core.transformer.TransformerConfig`) are extended with a `configure_model` method that defines how model weights are initialized and loaded in a way that is compliant with training via NeMo2.
- Various modifications and extensions to common PyTorch classes, such as adding a `MegatronDataSampler` (and re-sampler such as `PRNGResampleDataset` or `MultiEpochDatasetResampler`) to your `LightningDataModule`.

This tutorial demonstrates the construction of a simple Megatron model to classify MNIST digits.

We define a wrapper for the MNIST dataset that returns a dictionary instead of a tuple or a tensor.


First, we define a simple loss function. These should inherit from losses in nemo.lightning.megatron_parallel and can inherit from MegatronLossReduction.  The output of forward and backwared passes happen in parallel. The reduce function is required. It is only used for collecting forward output for inference, as well as for logging.

```
class SameSizeLossDict(TypedDict):
    """This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size."""

    avg: Tensor


class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: Dict[str, Tensor]) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        x_hat = forward_out["x_hat"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()
    class ClassifierLossReduction(MegatronLossReduction):
        """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

        def forward(self, batch: MnistItem, forward_out: Tensor) -> Tuple[Tensor, SameSizeLossDict]:
            """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

            Args:
                batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
                forward_out: the output of the forward method inside LitAutoEncoder.

            Returns:
                A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                    backpropagation and the ReductionT will be passed to the reduce method
                    (which currently only works for logging.).
            """
            digits = batch["label"]
            digit_logits = forward_out
            loss = nn.functional.cross_entropy(digit_logits, digits)
            return loss, {"avg": loss}

        def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
            """Works across micro-batches. (data on single gpu).

            Note: This currently only works for logging and this loss will not be used for backpropagation.

            Args:
                losses_reduced_per_micro_batch: a list of the outputs of forward

            Returns:
                A tensor that is the mean of the losses. (used for logging).
            """
            mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
            return mse_losses.mean()
```
```class MnistItem(TypedDict):
    data: Tensor
    label: Tensor
    idx: int

class MNISTCustom(MNIST):
    def __getitem__(self, index: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }
```
