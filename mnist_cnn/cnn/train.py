import functools as ft
import time

import equinox as eqx
import jax.numpy as jnp
import jax.random
import optax
from icecream import ic
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from model.model import Model
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def main():
    # Transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(trainset))
    validation_size = len(trainset) - train_size
    train_dataset, validation_dataset = random_split(
        trainset, [train_size, validation_size]
    )

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validationloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

    model = Model()
    optim = optax.adamw(learning_rate=0.001)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.key(45)
    start_time = time.time()
    for epoch in range(5):
        model = train(model, opt_state, optim, trainloader, key)
        eval_loss = evaluate(model, validationloader)
        ic(epoch, eval_loss)
    end_time = time.time()

    ic("training took " + str(end_time - start_time))
    eqx.tree_serialise_leaves("../models/model.eqx", model)


def loss_fn(
    model: PyTree,
    x: Float[Array, "batch 1 28 28"],
    y: Int[Array, " batch"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    partial_model = ft.partial(model, key=key)
    pred_y = eqx.filter_vmap(partial_model)(x)

    loss = optax.softmax_cross_entropy(pred_y, y)

    return jnp.mean(loss)


@eqx.filter_jit
def step(
    model: PyTree,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Array,
    y: Array,
    key: PRNGKeyArray,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def one_hot_encode(labels, num_classes=10):
    return jnp.eye(num_classes)[labels]


def evaluate(model: PyTree, eval_dataloader: DataLoader):
    loss = 0
    accuracy = 0
    counter = 0
    acc_fn = lambda a, b: jnp.argmax(a) == b
    jitted_loss = eqx.filter_jit(loss_fn)
    for x, y in eval_dataloader:
        counter += len(x)
        x = x.numpy()
        target = y.numpy()
        y = one_hot_encode(y.numpy())
        loss += jitted_loss(model, x, y, key=None)

        pt_model = ft.partial(model, key=None)
        output = eqx.filter_vmap(pt_model)(x)
        accuracy += jnp.sum(jax.vmap(acc_fn)(output, target))

    return loss / counter, accuracy / counter


def train(
    model: PyTree,
    opt_state: PyTree,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    key: PRNGKeyArray,
):
    for i, (x, y) in enumerate(train_dataloader):
        key, subkey = jax.random.split(key)
        x = x.numpy()
        y = one_hot_encode(y.numpy())
        model, opt_state, train_loss = step(model, optim, opt_state, x, y, subkey)
    return model


if __name__ == "__main__":
    main()
