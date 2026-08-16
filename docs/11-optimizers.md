# 11 — Optimizers

After `loss.wire()` has propagated gradients through the graph, every trainable
`Leaf` holds a direction in which its value would increase the loss. An
**optimizer** uses those gradients to change the parameters in the opposite
direction.

`Optimizer` holds the parameters it is responsible for. `SGD` is the first
concrete optimizer: it applies the same update to each parameter.

## Stochastic Gradient Descent

For one parameter $\theta$, with gradient $\partial L / \partial \theta$, SGD
updates it as

$$
\theta \leftarrow \theta - \eta\frac{\partial L}{\partial \theta}.
$$

Here $\eta$ is the `learning_rate`. The gradient points toward increasing loss,
so subtracting it makes a small move toward lower loss. A larger learning rate
moves faster but can overshoot; a smaller one is steadier but needs more steps.

The same rule applies element by element to arrays, so one `SGD` instance can
update a layer's weight matrix and bias vector together.

### Training loop
Give the optimizer the model's parameters once, before training:

```python
optimizer = SGD(model.parameters(), learning_rate=0.03)
```

Each step then has three parts:

```python
optimizer.zero_grad()
loss = ((model(inputs) - targets) ** 2).mean()
loss.wire()
optimizer.step()
```

`zero_grad()` clears the gradients left by the previous step. `wire()` computes
the new gradients, and `step()` changes the parameter data using them. Keeping
these jobs separate makes the training loop explicit: first measure the error,
then find its gradients, then update the model.
