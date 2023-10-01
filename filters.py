import pandas as pd
from scipy import ndimage

# uniform filter
def uniformfilter(df, size):
    df_filt_uni = ndimage.uniform_filter(df, size)
    df_filt_uni = pd.DataFrame(df_filt_uni)  # transform float in pandas dataframe
    return df_filt_uni

# gaussian filter
def gaussianfilter(df, size):
    df_filt_gau = ndimage.gaussian_filter(df, size)
    df_filt_gau = pd.DataFrame(df_filt_gau) # transform float in pandas dataframe
    return df_filt_gau

# median filter
def medianfilter(df, size):
    df_filt_med = ndimage.median_filter(df, size)
    df_filt_med = pd.DataFrame(df_filt_med) # transform float in pandas dataframe
    return df_filt_med

