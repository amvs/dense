from .morlet import morlet
from .yang import yang
def filter_bank(wavelet_name, max_scale, nb_orients, *args):
    """
    Returns a list of tensors, one per scale, for the selected wavelet.
    
    Args:
        wavelet_name (str): "morlet", ...
        max_scale (int): maximum scales
        nb_orients (int): total orientations
    """
    
    wavelets_map = {
        "morlet": morlet,
        "yang": yang,
    }
    
    if wavelet_name not in wavelets_map:
        raise ValueError(f"Unknown wavelet: {wavelet_name}")
    
    wavelet_func = wavelets_map[wavelet_name]
    return wavelet_func(max_scale, nb_orients, *args)



__all__ = ["filter_bank"]