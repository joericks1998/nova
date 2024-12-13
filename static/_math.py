#mean for evaluating forward pass
def mean(arr):
    return sum([i*arr[i] for i in range(0, len(arr))])
