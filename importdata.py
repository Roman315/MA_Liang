import pandas as pd

def import_data(path, n_resize):
    
    # read infos
    info = pd.read_csv(path, sep=',', header=None, names = [0,1,2], nrows = 13, encoding='latin1')

    # read height data
    df_roh = pd.read_csv(path, sep=',', header=None, skiprows=15, encoding='latin1') 
    df = df_roh.copy()
    
    df_columns = df.shape[1] #number of columns
    
    # loop to replace all , with . (column by column)
    i=0
    while i < df_columns:
        df[i] = df[i].astype(float) # replace , by . and transform str to float
        i=i+1
        
    # resize dataframe
    df = df.iloc[::n_resize, ::n_resize]
    data_size = df.size
    
    # extract necessary informations from info
    magnification = float(info.iloc[5,1])
    
    units_conversion = info.iloc[6,1]
    units_conversion = float(units_conversion) #convert str to float = units_conversion.str.replace(',','.').astype(float)
    
    return df, info, magnification, units_conversion, data_size
