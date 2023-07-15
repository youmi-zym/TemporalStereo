import torch
import torch.nn as nn

def timeTestTemplate(module, *args, **kwargs):
    """
    Module time test, inputs can be in tuple/list or dict
    Args:
        module:                         (nn.Module, callable Function): the module we want to time test
        *args:                          (tuple, list): the inputs of module
        **kwargs:                       (dict): the inputs of module

    Returns:
        avg_time:                       (double, float): the average time of inference for one round
    """
    device = kwargs.pop('device', None)
    assert device is not None, "param: device must be given, e.g., device=torch.device('cuda:0')"
    iters = kwargs.pop('iters', 1000)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    if isinstance(module, nn.Module):
        module.eval().to(device)

    avg_time = 0.0
    count = 0

    with torch.no_grad():
        for i in range(iters):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            if len(args) > 0:
                module(*args)
            if len(kwargs) > 0:
                module(**kwargs)
            end_time.record()
            torch.cuda.synchronize(device)
            if i >=100 and i < 900:
                avg_time += start_time.elapsed_time(end_time)
                count += 1
    avg_time = avg_time / count # in milliseconds
    avg_time = avg_time / 1000 # in second

    return avg_time
