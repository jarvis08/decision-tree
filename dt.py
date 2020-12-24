import sys
import numpy as np
#from prettyprinter import pprint
from collections import Counter


def load_data(path):
    objects = list()
    with open(path, "r") as f:
        read = f.readlines()
        attributes = read[0].split("\t")
        attributes[-1] = attributes[-1].replace("\n", '')
        for i in range(1, len(read)):
            splitted = read[i].split("\t")
            splitted[-1] = splitted[-1].replace("\n", '')
            objects.append(splitted)
    return attributes, objects


def save_results(path_out, attributes, objects):
    with open(path_out, 'w') as f:
        n_attr = len(attributes)
        for i in range(n_attr):
            if i < n_attr - 1:
                f.write(attributes[i] + '\t')
            else:
                f.write(attributes[i] + '\n')

        for obj in objects:
            for i in range(n_attr):
                if i < n_attr - 1:
                    f.write(obj[i] + '\t')
                else:
                    f.write(obj[i] + '\n')


def calculate_entropy(D_cls):
    counted = Counter(D_cls)
    total = sum(counted.values())
    classes = set(D_cls)
    entropy = 0
    for c in classes:
        p = float(counted[c]) / total
        entropy -= p * np.log2(p)
    return entropy


def check_majority(D_cls):
    counted = Counter(D_cls)
    max_cnt = -1
    max_cls = 0
    for k, v in counted.items():
        if max_cnt < v:
            max_cnt = v
            max_cls = k
    return max_cls


def get_indices(D_attr):
    # Make p dictionary per attribute
    attributes = set(D_attr)
    rst = []
    for attr in attributes:
        tmp = []
        for i in range(len(D_attr)):
            if D_attr[i] == attr:
                tmp.append(i)
        rst.append(tmp)
    return attributes, rst


def construct_tree(D, names):
    D_cls = D[:, -1]
    if len(names) < 2:
        return {"cls": str(check_majority(D_cls))}
    if len(set(D_cls)) == 1:
        return {"cls": str(D_cls[0])}

    attr_names = names[:-1]
    gains = np.array([])
    info_D = calculate_entropy(D_cls)
    for v in attr_names:
        D_v = D[:, attr_names.index(v)]
        _, indices = get_indices(D_v)

        if len(indices) < 2:
            continue

        info = 0
        for idx in indices:
            info += calculate_entropy(D_cls[idx]) * len(idx) / len(D)
        gains = np.append(gains, np.array([info_D - info]))
    if not gains.any():
        return {"cls": str(check_majority(D_cls))}
    selected = gains.argmax()
    best = attr_names[selected]

    sub_tree = {best: dict()}

    D_v = D[:, selected]
    values, indices = get_indices(D_v)

    names_nxt = np.delete(names, selected)
    names_nxt = names_nxt.tolist()
    D_new = np.delete(D, selected, axis=1)
    for v, idx in zip(values, indices):
        D_nxt = D_new[idx]
        sub_tree[best][str(v)] = construct_tree(D_nxt, names_nxt)
    sub_tree[best]["no_attr"] = str(check_majority(D_cls))
    return sub_tree


def test_dt(trained_model, label, path_in, path_out):
    attributes, objects = load_data(path_in)
    for i in range(len(objects)):
        inference = True
        subtree = trained_model
        while inference:
            node = list(subtree.keys())
            edge = subtree[node[0]]
            if node[0] == 'cls':
                objects[i].append(edge)
                inference = False
            else:
                value = objects[i][attributes.index(node[0])]
                # incase of absence of required branch
                if not edge.get(value):
                    value = "no_attr"
                    #value = list(edge.keys())[0]
                    objects[i].append(edge[value])
                    inference = False
                subtree = edge[value]
    attributes.append(label)
    save_results(path_out, attributes, objects)


def train_dt(path_in):
    att_names, objects = load_data(path_in)
    # make attributes' dict
    model = construct_tree(np.array(objects), att_names)
    print("Trained Model=====================================")
    #pprint(model)
    print("==================================================")
    return model, att_names[-1]
    

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 4:
        print("Not enough arguments given.")
        exit()

    TRAIN_FILENAME = argv[1]
    TEST_FILENAME = argv[2]
    OUTPUT_FILENAME = argv[3]
    print("Run decision tree algorithm with given arguments.")
    print("Training filename : {}".format(TRAIN_FILENAME))
    print("Test filename : {}".format(TEST_FILENAME))
    print("Output filename : {}".format(OUTPUT_FILENAME))

    bt_model, class_label = train_dt(TRAIN_FILENAME)
    test_dt(bt_model, class_label, TEST_FILENAME, OUTPUT_FILENAME)
