

class SoftmaxCrossEntropy(object):

    def __init__(self, average=True, reduce=True):
        self.average = average
        self.reduce = reduce

    def __call__(self, y, t):
        batch, num_class = y.data.shape
        loss = t * (- y + y.exp().sum(1).log().repeat(1, num_class).view(num_class, batch).transpose(0, 1))
        divider = batch if self.average is True else 1.0
        if self.reduce is True:
            return loss.sum() / divider
        else:
            return loss / divider


def softmax_cross_entropy(y, t, average=True, reduce=True):
    return SoftmaxCrossEntropy(average, reduce)(y, t)
