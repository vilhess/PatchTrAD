import json

def load_config(dataset, filename="config.json"):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config[dataset]

def load_results(filename="aucs.json"):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def save_results(filename, dataset, model, auc):
    results = load_results(filename)
    if dataset not in results:
        results[dataset]={}
    results[dataset][model] = auc
    with open(filename, "w") as f:
        json.dump(results, f)