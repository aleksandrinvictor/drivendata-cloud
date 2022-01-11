from importlib import import_module
from typing import Any, Callable, Dict, Union

import albumentations as A


def build_object(object_cfg: Dict[str, Any], **kwargs: Dict[str, Any]) -> Callable:

    if "class_name" not in object_cfg.keys():
        raise ValueError("class_name key schould be in config")

    if "params" in object_cfg.keys():
        params = object_cfg["params"]

        for key, val in params.items():
            kwargs[key] = val
    else:
        params = {}

    return get_instance(object_cfg["class_name"])(**kwargs)


def get_instance(object_path: str) -> Callable:

    module_path, class_name = object_path.rsplit(".", 1)
    module = import_module(module_path)

    return getattr(module, class_name)


def load_metrics(cfg: Dict[str, Any]) -> Union[Dict[str, Callable], None]:
    """Load metrics

    Parameters
    ----------
    cfg: DictConfig
        metrics config

    Returns
    -------
    Dict[str, Callable]
    """

    if cfg is None:
        return None

    metrics: Dict[str, Callable] = {}

    for a in cfg:

        if isinstance(a, dict) and "params" in a.keys():
            params: Dict[str, Any] = {k: (v if type(v) != list else tuple(v)) for k, v in a["params"].items()}
        else:
            params = {}

        metric = get_instance(a["class_name"])(**params)  # type: ignore

        metrics[a["name"]] = metric  # type: ignore

    return metrics


def load_augs(cfg: Dict[str, Any]) -> A.Compose:
    """Load albumentations

    Parameters
    ----------
    cfg: DictConfig
        augmentations config

    Returns
    -------
    A.Compose
    """
    augs = []
    for a in cfg:
        if a["class_name"] == "albumentations.OneOf":  # type: ignore
            small_augs = []
            for small_aug in a["transforms"]:  # type: ignore
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (v if not isinstance(v, list) else tuple(v))
                    for k, v in small_aug["params"].items()  # type: ignore
                }

                aug = get_instance(small_aug["class_name"])(**params)  # type: ignore
                small_augs.append(aug)

            if "params" in a.keys():  # type: ignore
                params = {k: (v if type(v) != list else tuple(v)) for k, v in a["params"].items()}  # type: ignore
            aug = get_instance(a["class_name"])(small_augs, **params)  # type: ignore
            augs.append(aug)

        else:
            if "params" in a.keys():  # type: ignore
                params = {k: (v if type(v) != list else tuple(v)) for k, v in a["params"].items()}  # type: ignore
            else:
                params = {}
            aug = get_instance(a["class_name"])(**params)  # type: ignore
            augs.append(aug)

    return A.Compose(augs)
