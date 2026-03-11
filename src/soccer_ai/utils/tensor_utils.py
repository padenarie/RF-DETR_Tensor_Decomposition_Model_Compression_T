import numpy as np
import tntorch_pierre as tnp

def metrics(tensor_original,tt_tensor, verbose=False):
    tt_numel = sum(core.numel() for core in tt_tensor.cores)
    compression_ratio = tensor_original.numel() / tt_numel
    relative_error = tnp.relative_error(tensor_original, tt_tensor)
    RMSE = tnp.rmse(tensor_original, tt_tensor)
    R2 = tnp.r_squared(tensor_original, tt_tensor)
    if verbose:
        print(tt_tensor)
        print('Core ranks:', tt_tensor.ranks_tt)
        print('Total number of elements in TT cores:', tt_numel)
        print('Compression ratio: {}/{} = {:g}'.format(tensor_original.numel(), tt_numel, compression_ratio))
        print('Relative error:', relative_error)
        print('RMSE:', RMSE)
        print('R^2:', R2)
        
    return compression_ratio, relative_error, RMSE, R2