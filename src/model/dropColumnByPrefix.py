
def dropColumnByPrefix(df, prefix):
    prefix = prefix + "_"
    df.drop(list(df.filter(regex=prefix)), axis=1, inplace=True)
    return df
