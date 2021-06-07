
def CreateDataLoader(_root, _list_dir):
    data_loader = None

    from data.aligned_data_loader import AlignedDataLoader
    data_loader = AlignedDataLoader(_root, _list_dir)

    # if opt.align_data > 0:
    #     from data.aligned_data_loader import AlignedDataLoader
    #     data_loader = AlignedDataLoader()
    # else:
    #     from data.unaligned_data_loader import UnalignedDataLoader
    #     data_loader = UnalignedDataLoader()
    # print(data_loader.name())
    # data_loader.initialize(opt)
    return data_loader

def CreateDataLoaderIIW(_root, _list_dir, mode, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import IIWDataLoader
    data_loader = IIWDataLoader(_root, _list_dir, mode, batch_size)

    return data_loader


def CreateDataLoaderSAW(_root, _list_dir, mode, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import SAWDataLoader
    data_loader = SAWDataLoader(_root, _list_dir, mode, batch_size=batch_size)
    return data_loader


def CreateDataLoaderRender(_root, _list_dir, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import RenderDataLoader
    data_loader = RenderDataLoader(_root, _list_dir, batch_size)

    return data_loader


def CreateDataLoaderIIWTest(_root, _list_dir, mode, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import IIWTESTDataLoader
    data_loader = IIWTESTDataLoader(_root, _list_dir, mode, batch_size)

    return data_loader


def CreateDataLoaderOpenSurfaces(_root, _list_dir, mode, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import OpenSurface_DataLoader
    data_loader = OpenSurface_DataLoader(_root, _list_dir, mode, batch_size)
    return data_loader


def CreateDataLoaderCGIntrinsics(_root, _list_dir, batch_size=16):
    data_loader = None
    from data.aligned_data_loader import CGIntrinsics_DataLoader
    data_loader = CGIntrinsics_DataLoader(_root, _list_dir, batch_size)
    return data_loader


def CreateDataLoader_TEST(_root, _list_dir):
    data_loader = None
    from data.aligned_data_loader import AlignedDataLoader_TEST
    data_loader = AlignedDataLoader_TEST(_root, _list_dir)

    # if opt.align_data > 0:
    #     from data.aligned_data_loader import AlignedDataLoader
    #     data_loader = AlignedDataLoader()
    # else:
    #     from data.unaligned_data_loader import UnalignedDataLoader
    #     data_loader = UnalignedDataLoader()
    # print(data_loader.name())
    # data_loader.initialize(opt)
    return data_loader
