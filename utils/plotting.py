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
def to_colors(data: np.ndarray, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a numpy array to an array of colors with a given colormap
    """
    
    if vmin is None:
        vmin = data.min()

    if vmax is None:
        vmax = data.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    def to_string(r, g, b):
        """
        Convert an (r, g, b) triplet in the range 0-1 to a hex color string
        """
        return f'#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}'

    return np.array([to_string(r, g, b) for (r, g, b, _) in m.to_rgba(data.ravel())]).reshape(data.shape)

