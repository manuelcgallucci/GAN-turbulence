import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(dict_):
    scales = len(dict_["meanS2"])
    arr_means = np.zeros((3, scales))
    arr_std = np.zeros((3, scales))
    arr_means[0,:] = dict_["meanS2"]
    arr_means[1,:] = dict_["meanSkew"]
    arr_means[2,:] = dict_["meanFlatness"]
    arr_std[0,:] = dict_["stdS2"]
    arr_std[1,:] = dict_["stdSkew"]
    arr_std[2,:] = dict_["stdFlatness"]
    return arr_means, arr_std

def plot_bars(metrics, names):
    base_colors = np.array([[70, 108, 184], [64, 173, 88], [245, 190, 95]]) / 255.0
    detail_colors = np.array([[44, 91, 184], [34, 163, 63], [247, 172, 42]]) / 255.0
    
    X = np.arange(len(names))
    fig, ax = plt.subplots()

    ax.bar(X + 0.00, [metric[0] for metric in metrics], color=base_colors[0], width=0.25, label="MSE s2")
    ax.bar(X + 0.25, [metric[1] for metric in metrics], color=base_colors[1], width=0.25, label="MSE Skewness")
    ax.bar(X + 0.50, [metric[2] for metric in metrics], color=base_colors[2], width=0.25, label="MSE Flatness")
    
    ax.scatter(X + 0.00, [metric[3] for metric in metrics], color=detail_colors[0], label="MSE s2 k ")
    ax.scatter(X + 0.25, [metric[4] for metric in metrics], color=detail_colors[1], label="MSE Skewness k")
    ax.scatter(X + 0.50, [metric[5] for metric in metrics], color=detail_colors[2], label="MSE Flatness k")
    
    ax.set_xticks(X+0.25)
    ax.set_xticklabels(names, rotation=30)
    ax.legend()
    ax.set_title("Metrics for all models")
    plt.savefig("./bars.png")
    plt.close()

def generate_metrics(struct_mean1, struct_mean2):
    metrics = np.mean(np.square(struct_mean1 - struct_mean2), axis=1)

    c = np.mean((struct_mean1 - struct_mean2), axis=1)
    metrics_c = np.mean(np.square(struct_mean1 - struct_mean2 - c[:,None]), axis=1)
    return np.concatenate((metrics, metrics_c), axis=0)

def main():
    json_file = "output.json"

    with open(json_file, 'r') as f:
        data_dict = json.load(f)

    struct_mean_real, struct_std_real = load_data(data_dict["data"])
    metrics = []
    model_hashes = []
    for model_hash in data_dict.keys():
        if model_hash == "data":
            pass
        else:
            struct_mean_generated, struct_std_generated = load_data(data_dict[model_hash])

            metric = generate_metrics(struct_mean_real, struct_mean_generated)
            metrics.append(metric)
            model_hashes.append(model_hash)

    plot_bars(metrics, model_hashes)



if __name__ == "__main__":
    main()