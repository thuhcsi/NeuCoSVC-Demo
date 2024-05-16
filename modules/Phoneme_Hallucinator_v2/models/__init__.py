
def get_model(hps):
    if hps.model == 'pc_acset_vae':
        from .pc_acset_vae import ACSetVAE
        model = ACSetVAE(hps)
    else:
        raise ValueError()

    return model
