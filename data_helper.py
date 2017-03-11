import numpy as np

def load_data(path,delimiter,label_last,header,y_values):

    #x = [line.split(delimiter)[1:] for line in open(path, "r").readlines()]
    x = []
    sorted(y_values)
    i = 0
    for line in open(path, "r").readlines():
        if header and i == 0:
            i = i + 1
            continue
        if label_last:
            z = line.split(delimiter)[:-1]
        else:
            z = line.split(delimiter)[1:]
        i = i + 1
        x.append(z)
    y = []
    i = 0
    for line in open(path, "r").readlines():
        if header and i == 0:
            i = i + 1
            continue
        if label_last:
            label = line.split(delimiter)[-1]
        else:
            label = line.split(delimiter)[0]
        i = i + 1
        y_vec = [0]*len(y_values)
        #print(label)
        y_vec[y_values.index(label)] = 1
        y.append(y_vec)
    return [np.array(x), np.array(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch generator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter_gen(path, batch_size, num_epochs, label_last, header, delimiter, y_values):
    """
    Generates a batch generator for a dataset.
    """
    for epoch in range(num_epochs):
        x = []
        y = []
        i = 0
        for line in open(path):
            if header and i == 0:
                i = i + 1
                continue
            if label_last:
                z = line.split(delimiter)[:-1]
                label = line.split(delimiter)[-1]
            else:
                z = line.split(delimiter)[1:]
                label = line.split(delimiter)[0]
            x.append(z)
            y_vec = [0]*len(y_values)
            #print(label)
            y_vec[y_values.index(label)] = 1
            y.append(y_vec)


            if i%batch_size == batch_size - 1:
                yield zip(x,y)
                x = []
                y = []
            i = i + 1
        if (i-1)%batch_size != batch_size - 1:
            yield zip(x,y)
