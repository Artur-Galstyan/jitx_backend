from typing import Optional

import equinox as eqx
import jax.random
from jaxtyping import Array, PRNGKeyArray


class Model(eqx.Module):
    conv: eqx.nn.Conv2d
    mlp: eqx.nn.MLP
    max_pool: eqx.nn.MaxPool2d

    dropout: eqx.nn.Dropout

    def __init__(self):
        key, *subkeys = jax.random.split(jax.random.PRNGKey(33), 6)
        self.conv = eqx.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=2, key=subkeys[1]
        )
        self.max_pool = eqx.nn.MaxPool2d(kernel_size=2)
        self.mlp = eqx.nn.MLP(
            in_size=2028, out_size=10, depth=3, width_size=128, key=subkeys[0]
        )
        self.dropout = eqx.nn.Dropout()

    def __call__(self, x: Array, key: Optional[PRNGKeyArray]):
        inference = True if key is None else False
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.dropout(x, key=key, inference=inference)
        x = x.ravel()
        x = self.mlp(x)
        x = jax.nn.softmax(x)
        return x
