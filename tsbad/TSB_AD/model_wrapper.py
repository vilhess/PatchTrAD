def run_PatchTrAD(data_train, data, win_size=10, lr=1e-3, patch_len=8, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256):
    from PatchTrAD import PatchTrAD
    clf = PatchTrAD(win_size=win_size, feats=data.shape[1], lr=lr, patch_len=patch_len, stride=stride, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()