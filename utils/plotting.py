import numpy as np
import pandas as pd
import matplotlib as mpl


def handle_types(f):
    """
    Decorator to allow the to_colors() function to handle list and
    pandas DataFrame/Series data as well as numpy arrays. 
    """
    
    def _to_colors(data: np.ndarray | pd.DataFrame | pd.Series, *args, **kwargs):
        
        if isinstance(data, pd.DataFrame):
            
            out = f(data.values, *args, **kwargs)
            
            return pd.DataFrame(out, columns=data.columns, index=data.index)
        
        if isinstance(data, pd.Series):
            
            out = f(data.values, *args, **kwargs)
            
            return pd.Series(out, index=data.index)
        
        if isinstance(data, list):
            
            out = f(np.array(data), *args, **kwargs)
            
            return out.tolist()
        
        return  f(data, *args, **kwargs)
            
    return _to_colors

@handle_types
def to_colors(data: np.ndarray, cmap='viridis', vmin=None, vmax=None, alpha=1.0, nan_color: str='#b2b2b2'):
    """
    Convert a numpy array to an array of colors with a given colormap
    """

    # bound alpha
    if alpha < 0:
        alpha = 0
    if alpha > 1: 
        alpha = 1

    # add alpha to nan_color if not present
    if alpha is not None:
        if len(nan_color) == 7:
            nan_color += f'{int(alpha * 255):02X}'

    # temporarily replace nans with 0, then fill with nan_color at the end
    nan_inds = np.isnan(data)

    if nan_inds.any():
        data[nan_inds] = 0
        nans = True

    else:
        nans = False
        
    if vmin is None:
        vmin = np.nanmin(data)

    if vmax is None:
        vmax = np.nanmax(data)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    def to_string(r, g, b, a):
        """
        Convert an (r, g, b) triplet in the range 0-1 to a hex color string
        """
        if a == 1.0:
            return '#' + ''.join([f'{int(c * 255):02X}' for c in [r, g, b]])
        
        else:
            return '#' + ''.join([f'{int(c * 255):02X}' for c in [r, g, b, a]])
        
    out = np.array([to_string(r, g, b, alpha) for (r, g, b, _) in m.to_rgba(data.ravel())]).reshape(data.shape)

    if nans:
        out[nan_inds] = nan_color

    data[nan_inds] = np.nan

    return out



