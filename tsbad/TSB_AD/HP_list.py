Multi_algo_HP_dict = {
    'PatchTrAD': {
        'win_size': [50, 100, 150],
        'lr': [0.0001, 0.001],
        'patch_len': [8, 16],
        'stride': [6, 12],
        'd_model': [128, 256],
        'n_heads': [4, 8],
        'n_layers': [3, 6],
        'd_ff': [256, 512],
    },

}

Optimal_Multi_algo_HP_dict = {
    'PatchTrAD': {'win_size': 50, 'lr': 0.0001, 'patch_len': 16, 'stride': 12, 'd_model': 128, 'n_heads': 4, 'n_layers': 3, 'd_ff': 512},
}


Uni_algo_HP_dict = {
    'PatchTrAD': {
        'win_size': [50, 100, 150],
        'lr': [0.0001, 0.001],
        'patch_len': [8, 16],
        'stride': [6, 12],
        'd_model': [128, 256],
        'n_heads': [4, 8],
        'n_layers': [3, 6],
        'd_ff': [256, 512],
    },
}

Optimal_Uni_algo_HP_dict = {
    'PatchTrAD': {'win_size': 50, 'lr': 0.0001, 'patch_len': 16, 'stride': 12, 'd_model': 128, 'n_heads': 4, 'n_layers': 3, 'd_ff': 512},
}