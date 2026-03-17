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


def bytes_tensor(t):
    return t.numel() * t.element_size()

def full_model_size_bytes(
    model_input,
    include_tt_cores=True,
    reference_bytes=None,   # pass baseline (usually pre-TT size of SAME model)
    verbose=False,
):
    net = model_input.model.model
    total = 0

    # Registered params + buffers
    for p in net.parameters():
        total += bytes_tensor(p)
    for b in net.buffers():
        total += bytes_tensor(b)

    # TT cores may not be registered as params
    tt_extra = 0
    if include_tt_cores:
        for m in net.modules():
            if hasattr(m, "tt_tensor") and hasattr(m.tt_tensor, "cores"):
                for c in m.tt_tensor.cores:
                    tt_extra += bytes_tensor(c)

    total_with_tt = total + tt_extra

    compression_rate = None
    if reference_bytes is not None and total_with_tt > 0:
        compression_rate = reference_bytes / total_with_tt

    if verbose:
        print(f"registered params+buffers: {total} bytes")
        print(f"extra TT cores:            {tt_extra} bytes")
        print(f"total:                     {total_with_tt} bytes")
        if compression_rate is not None:
            print(f"total compression rate:    {compression_rate:.4f}x")

    return {
        "total_bytes": total_with_tt,
        "registered_bytes": total,
        "tt_extra_bytes": tt_extra,
        "compression_rate": compression_rate,
    }