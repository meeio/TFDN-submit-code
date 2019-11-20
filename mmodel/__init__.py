
def get_module(name):
    if name == 'DANN':
        from .DANN import DANN_model 
        return DANN_model.params, DANN_model.DANNModule()
    elif name == 'DADA':
        from .DADA import DADA_model 
        return DADA_model.params, DADA_model.DADAModule()
    elif name == 'DEMO':
        from .DEMO import DEMO_model
        return DEMO_model.params, DEMO_model.DEMOModel()
    elif name == 'DEMO1':
        from .DEMO1 import DEMO1_model
        return DEMO1_model.params, DEMO1_model.DEMO1Model()
    elif name == 'DEMO11':
        from .DEMO11 import DEMO1_model
        return DEMO1_model.params, DEMO1_model.DEMO1Model()