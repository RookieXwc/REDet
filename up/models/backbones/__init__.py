from .mobilenet_v2 import mobilenetv2  # noqa: F401
from .mobilenet_v3 import mobilenetv3  # noqa: F401
from .resnet import (resnet101,  # noqa
                     resnet152,
                     resnet18,
                     resnet34,
                     resnet50,
                     resnet_custom)

from .efficientnet import (  # noqa: F401
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
