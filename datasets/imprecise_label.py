import numpy as np
import torch
import pickle

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

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def get_hierarchical_partial_labels(samples, target, num_classes=100, partial_ration=0.5):
    samples, targets = np.array(samples), np.array(target)
    num_samples = targets.shape[0]

    meta = unpickle('data/cifar-100-python/meta')
    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}
    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
    fish#aquarium fish, flatfish, ray, shark, trout
    flowers#orchid, poppy, rose, sunflower, tulip
    food containers#bottle, bowl, can, cup, plate
    fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
    household electrical devices#clock, keyboard, lamp, telephone, television
    household furniture#bed, chair, couch, table, wardrobe
    insects#bee, beetle, butterfly, caterpillar, cockroach
    large carnivores#bear, leopard, lion, tiger, wolf
    large man-made outdoor things#bridge, castle, house, road, skyscraper
    large natural outdoor scenes#cloud, forest, mountain, plain, sea
    large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals#fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates#crab, lobster, snail, spider, worm
    people#baby, boy, girl, man, woman
    reptiles#crocodile, dinosaur, lizard, snake, turtle
    small mammals#hamster, mouse, rabbit, shrew, squirrel
    trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    vehicles 1#bicycle, bus, motorcycle, pickup truck, train
    vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''
    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]

        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i
    partial_targets = np.zeros((num_samples, num_classes))
    partial_targets[np.arange(num_samples), targets] = 1.0
    transition_matrix = np.eye(num_classes)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_ration
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    random_n = np.random.uniform(0, 1, size=(num_samples, num_classes))
    for j in range(num_samples):  # for each instance
        partial_targets[j, :] = (random_n[j, :] < transition_matrix[targets[j], :]) * 1

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