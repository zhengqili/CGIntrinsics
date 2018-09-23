import torch


def create_model(opt):
    model = None
    from .intrinsic_model import Intrinsics_Model
    model = Intrinsics_Model(opt)
    print("model [%s] was created" % (model.name()))

    # model.initialize()
    return model
