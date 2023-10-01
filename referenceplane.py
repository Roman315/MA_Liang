import pandas as pd

def upper_ref(df, prozent_ref):
    df_long = df.unstack().reset_index() # from 768x1024 zu 786432x3 Matrix
    df_long.columns = ["X","Y","Z"] # rename columns 
    
    columns = df_long.shape[0]
    num_pro = int(round(columns*prozent_ref)) # 
    
    df_ref_u = df_long.nlargest(num_pro,'Z')
    ref_u = df_ref_u.mean()
    
    return ref_u
    
def lower_ref(df, prozent_ref):
    df_long = df.unstack().reset_index() # from 768x1024 zu 786432x3 Matrix
    df_long.columns = ["X","Y","Z"] # rename columns 
    
    columns = df_long.shape[0]
    num_pro = int(round(columns*prozent_ref))
    
    df_ref_l = df_long.nsmallest(num_pro,'Z')
    ref_l = df_ref_l.mean()
    
    return ref_l