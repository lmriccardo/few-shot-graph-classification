import math


def glorot(tensor):
    """Apply the Glorot NN initialization (also called Xavier)"""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    """Fill a tensor with zeros if it is not Null"""
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
