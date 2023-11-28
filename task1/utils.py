import numpy as np

def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten a nested dictionary and combine nested keys using the provided separator.
    """
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    items = {k:v if not isinstance(v, list) else "; ".join(v) for k,v in items.items()}
    return items

def unflatten_dict(d, sep="/"):
    """
    Convert a flattened dictionary into a nested dictionary.
    """
    def set_nested_item(dic, keys, value):
        """Set item in nested dictionary."""
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    nested_dict = {}
    for k,v in d.items():
        if isinstance(v, str) and '; ' in v:
            d[k] = v.split('; ')
    for key, value in d.items():
        keys = key.split(sep)
        set_nested_item(nested_dict, keys, value)
        
    return nested_dict

def round_prediction(pred):
    return np.floor(pred+0.5)
