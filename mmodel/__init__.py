
def get_module(name):
    if name == 'prototype':
        from .prototype import params, model
        return params.get_params(), model.ProtopyteNetwork()
    elif name == 'DAN':
        from .deep_adaptation import params, dan_model
        return params.get_params(), dan_model.DeepAdaptationNetworks()
    elif name == 'TPN':
        from .transferable_prototype_network import tpn_params, tpn_model
        return tpn_params.get_params(), tpn_model.TransferableProtopyteNetwork()
    elif name == 'ALEX':
        from .alex import alex_finetune_params, alex_finetune_module
        return alex_finetune_params.get_params(), alex_finetune_module.AlexFinetune()
    elif name == 'Center_Loss_Toy':
        from .center_loss import toy_model, toy_params 
        return toy_params.params, toy_model.CenterLossToy()