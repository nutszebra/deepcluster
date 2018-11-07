

class SoftmaxCrossEntropy(object):

    def __init__(self, average=True, reduce=True, eps=1.0e-10):
        self.eps = eps
        self.average = average
        self.reduce = reduce

    def __call__(self, y, t):
        batch, num_class = y.data.shape
        loss = - t * torch.log(F.softmax(y, 1) + self.eps)
        divider = batch if self.average is True else 1.0
        if self.reduce is True:
            return loss.sum() / divider
        else:
            return loss / divider


def softmax_cross_entropy(y, t, average=True, reduce=True):
    return SoftmaxCrossEntropy(average, reduce)(y, t)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    w = torch.nn.Parameter(torch.FloatTensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.1, 0.1, 0.1]]))
    t = torch.FloatTensor([[0.2, 0.1, 0.1, 0.5, 0.1], [0.2, 0.1, 0.1, 0.5, 0.1]])
    optimizer = torch.optim.SGD([w], lr=0.01)
    for i in range(20000):
        loss = softmax_cross_entropy(w, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(F.softmax(w))
    print(t)
