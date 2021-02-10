from lucent.optvis import objectives
import torch

@objectives.wrap_objective()
def neuron(layer, n_channel, offset=(0, 0), batch=None):
    """Visualize a single neuron of a single channel.
    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.
    Odd width & height:               Even width & height:
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+
    """
    @objectives.handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        x, y = layer_t.shape[-1] // 2, layer_t.shape[-2] // 2
        return -layer_t[:, n_channel, :, y+offset[1], x+offset[0]].mean()
    return inner


@objectives.wrap_objective()
def slow(layer, decay_ratio=2):
    """Encourage neighboring images to be change slowly with L2 penalty
    """
    def inner(model):
        layer_t = model(layer)
        return (torch.mean(torch.sum((layer_t[:-1, ...] - layer_t[1:, ...]) ** 2, axis=0) + 
                (layer_t[0, ...] - layer_t[-1, ...]) ** 2))
    return inner

@objectives.wrap_objective()
def tv_slow(layer, decay_ratio=2):
    """Encourage neighboring images to be change slowly with L1 penalty
    """
    def inner(model):
        layer_t = model(layer)
        return (torch.mean(torch.sum(abs(layer_t[:-1, ...] - layer_t[1:, ...]), axis=0) + 
                abs(layer_t[0, ...] - layer_t[-1, ...])))
    return inner


@objectives.wrap_objective()
def intensity_preservation(layer, block_size, input_size):
    """Encourage neighboring images to change slowly by lying on the optic flow
    """
    def inner(model):
        penalty = 0

        layer_t = model(layer)
        for i in range(1, layer_t.shape[0] - 1):
            for k in range(0, input_size - block_size + 1, block_size):
                for j in range(0, input_size - block_size + 1, block_size):
                    rgx = slice(j+1, j+block_size-1)
                    rgy = slice(k+1, k+block_size-1)
                    dx = layer_t[i, :, rgy, (j+2):(j+block_size)] - layer_t[i, :, rgy, (j):(j+block_size-2)]
                    dy = layer_t[i, :, (k+2):(k+block_size), rgx] - layer_t[i, :, (k):(k+block_size-2), rgx]
                    ip = (i + 1) %  layer_t.shape[0]
                    im = (i - 1) %  layer_t.shape[0]
                    dt = layer_t[ip, :, rgy, rgx] - layer_t[im, :, rgy, rgx]

                    A = torch.stack([dx.reshape(-1), 
                                   dy.reshape(-1)], axis=1)
                    b = -dt.reshape(-1, 1)

                    M = torch.inverse(torch.matmul(A.T, A))

                    bP = torch.matmul(torch.matmul(A, M), torch.matmul(A.T, b))
                    delta_brightness = ((bP.view(-1) - b.view(-1)) ** 2).mean()
                    penalty += delta_brightness

        return penalty
    return inner