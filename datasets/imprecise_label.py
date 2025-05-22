import numpy as np
import torch

def get_cifar10_asym_noisy_labels(samples, targets, num_classes=10, noise_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)

    noise_idx = []
    noisy_targets = np.copy(targets)
    for i in range(num_classes):
        indices = np.where(targets == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < noise_ratio * len(indices):
                noise_idx.append(idx)
                # truck -> automobile
                if i == 9:
                    noisy_targets[idx] = 1
                # bird -> airplane
                elif i == 2:
                    noisy_targets[idx] = 0
                # cat -> dog
                elif i == 3:
                    noisy_targets[idx] = 5
                # dog -> cat
                elif i == 5:
                    noisy_targets[idx] = 3
                # deer -> horse
                elif i == 4:
                    noisy_targets[idx] = 7
    return noise_idx, samples, noisy_targets


def get_cifar100_asym_noisy_labels(samples, targets, num_classes=100, noise_ratio=0.5):
    p = np.eye(num_classes)
    num_superclasses = 20
    num_subclasses = 5
    samples, targets = np.array(samples), np.array(targets)


    def build_for_cifar100(size, noise):
        p = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        p[cls1, cls2] = noise
        p[cls2, cls1] = noise
        p[cls1, cls1] = 1.0 - noise
        p[cls2, cls2] = 1.0 - noise
        return p

    if noise_ratio > 0.0:
        for i in np.arange(num_superclasses):
            init, end = i * num_subclasses, (i+1) * num_subclasses
            p[init:end, init:end] = build_for_cifar100(num_subclasses, noise_ratio) 

        noise_idx = []
        noisy_targets = np.copy(targets)
        for idx in np.arange(noisy_targets.shape[0]):
            y = targets[idx]
            flipped = np.random.multinomial(1, p[y, :], 1)[0]
            noisy_targets[idx] = np.where(flipped == 1)[0]
            if noisy_targets[idx] != targets[idx]:
                noise_idx.append(idx)
    
    return noise_idx, samples, noisy_targets


def get_partial_labels(samples, targets, num_classes=10, partial_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)
    num_samples = targets.shape[0]

    partial_targets = np.zeros((num_samples, num_classes))
    partial_targets[np.arange(num_samples), targets] = 1.0

    transition_matrix =  np.eye(num_classes)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_ratio

    random_n = np.random.uniform(0, 1, size=(num_samples, num_classes))

    for j in range(num_samples):  # for each instance
        partial_targets[j, :] = (random_n[j, :] < transition_matrix[targets[j], :]) * 1
    
    # labels are one-hot
    return samples, partial_targets


def get_sym_noisy_labels(samples, targets, num_classes=10, noise_ratio=0.5):
    samples, targets = np.array(samples), np.array(targets)

    noise_idx = []
    noisy_targets = np.copy(targets)
    indices = np.random.permutation(len(samples))
    for i, idx in enumerate(indices):
        if i < noise_ratio * len(samples):
            noise_idx.append(idx)
            noisy_targets[idx] = np.random.randint(num_classes, dtype=np.int32)

    return noise_idx, samples, noisy_targets


def get_semisup_labels(samples, targets, num_classes=10, num_labels=400, include_lb_to_ulb=True):
    samples, targets = np.array(samples), np.array(targets)

    lb_samples_per_class = [int(num_labels / num_classes)] * num_classes

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        ulb_idx.extend(idx[lb_samples_per_class[c]:])
    
    if include_lb_to_ulb and len(lb_idx) < len(samples):
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    else:
        ulb_idx = lb_idx

    lb_samples, lb_targets = samples[lb_idx], targets[lb_idx]
    ulb_samples, ulb_targets = samples[ulb_idx], targets[ulb_idx]
    return lb_samples, lb_targets, ulb_samples, ulb_targets