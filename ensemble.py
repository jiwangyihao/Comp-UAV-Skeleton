import argparse
import pickle
import os
from cProfile import label

import numpy as np
from tqdm import tqdm
from skopt import gp_minimize


def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r = results[0][i][1] * weights[0]
        for j in range(1, len(models)):
            r += results[j][i][1] * weights[j]
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print("Accuracy: ", acc)
    print("Weights: ", weights)
    return -acc  # We want to maximize accuracy, hence minimize -accuracy


if __name__ == "__main__":
    label = np.load('./Skeleton-MixFormer/data/uav/test_label.npy')

    models = [
        "Skeleton-MixFormer/work_dir/uav/skmixf_b",
        "Skeleton-MixFormer/work_dir/uav/skmixf_j",
        "Skeleton-MixFormer/work_dir/uav/skmixf_bm",
        "Skeleton-MixFormer/work_dir/uav/skmixf_jm",
        "FR-Head/results/uav/b",
        "FR-Head/results/uav/j",
        "FR-Head/results/uav/bm",
        "FR-Head/results/uav/jm",
        "SiT-MLP/work_dir/uav/j",
        "SiT-MLP/work_dir/uav/b",
        "SiT-MLP/work_dir/uav/jm",
        "SiT-MLP/work_dir/uav/bm"
    ]
    tests = [
        "Skeleton-MixFormer/work_dir/test/test_b",
        "Skeleton-MixFormer/work_dir/test/test_j",
        "Skeleton-MixFormer/work_dir/test/test_bm",
        "Skeleton-MixFormer/work_dir/test/test_jm",
        "FR-Head/results/test/test_b",
        "FR-Head/results/test/test_j",
        "FR-Head/results/test/test_bm",
        "FR-Head/results/test/test_jm",
        "SiT-MLP/work_dir/test/test_j",
        "SiT-MLP/work_dir/test/test_b",
        "SiT-MLP/work_dir/test/test_jm",
        "SiT-MLP/work_dir/test/test_bm"
    ]

    results = []
    preds = []

    print("Loading models score...")

    for dir_path in models:
        with open(os.path.join(dir_path, 'epoch1_test_score.pkl'), 'rb') as result:
            results.append(list(pickle.load(result).items()))
            print("Loaded:", os.path.join(dir_path, 'epoch1_test_score.pkl'))

    print("Loading preds...")
    
    for dir_path in tests:
        with open(os.path.join(dir_path, 'epoch1_test_pred.npy'), 'rb') as pred:
            preds.append(np.load(pred))
            print("Loaded:", os.path.join(dir_path, 'epoch1_test_pred.npy'))

    print("Optimization on Test_A: ")

    space = [(0, 1.2) for i in range(len(models))]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))

    print("Merging pred on Test_B using weights...")

    merged_pred = preds[0] * result.x[0]

    for i in range(1, len(models)):
        merged_pred += preds[i] * result.x[i]

    print("Merged pred shape: ", merged_pred.shape)

    np.save("pred.npy", merged_pred)
