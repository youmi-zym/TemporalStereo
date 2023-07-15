from detectron2.utils.registry import Registry

AGGREGATION_REGISTRY = Registry("AGGREGATION")
AGGREGATION_REGISTRY.__doc__ = """
Registry for cost aggregation, which estimate aggregated cost volume from images
The registered object must be a callable that accepts one arguments:
1. A :class:`detectron2.config.CfgNode`
Registered object must return instance of :class:`nn.Module`.
"""


def build_aggregation(cfg):
    """
    Build a cost aggregation predictor from `cfg.MODEL.AGGREGATION.NAME`.
    Returns:
        an instance of :class:`nn.Module`
    """

    aggregation_name = cfg.MODEL.AGGREGATION.NAME
    aggregation_predictor = AGGREGATION_REGISTRY.get(aggregation_name)(cfg)
    return aggregation_predictor