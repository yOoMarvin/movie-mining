def addPrefixToColumn(df, prefix):
    """
    adds a prefix to the oneHotEncoded DF
    :param df: oneHotEncoded DataFrame
    :param prefix: prefix to add to the column name
    :return: DataFrame with changed columnnames
    """

    
    """
    for column in df:
        if(column != "id:"):
            df = df.rename(columns={column: prefix+"_"+str(column)})
    """
    df = df.add_prefix(prefix+"_")
    return df