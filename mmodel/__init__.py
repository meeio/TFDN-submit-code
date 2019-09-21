
def get_module(name):
    if name == 'DANN':
        from .DANN import DANN_model 
        return DANN_model.params, DANN_model.DANNModule()
    elif name == 'DADA':
        from .DADA import DADA_model 
        return DADA_model.params, DADA_model.DADAModule()