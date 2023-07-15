from detectron2.utils.registry import Registry


PREDICTION_REGISTRY = Registry("PREDICTION")
PREDICTION_REGISTRY.__doc__ = """
Registry for preditions, which predict disparity maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
Registered object must return instance of :class:`nn.Module`.
"""


def build_prediction(cfg):
    """
    Build a prediction from `cfg.MODEL.PREDICTION.NAME`.
    Returns:
        an instance of :class:`nn.Module`
    """

    prediction_name = cfg.MODEL.PREDICTION.NAME
    prediction = PREDICTION_REGISTRY.get(prediction_name)(cfg)
    return prediction


