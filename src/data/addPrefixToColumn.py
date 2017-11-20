def addPrefixToColumn(df, prefix):
    """
    adds a prefix to the oneHotEncoded DF
    :param df: oneHotEncoded DataFrame
    :param prefix: prefix to add to the column name
    :return: DataFrame with changed columnnames
    """
    i = 0

    for column in df:
        if(column != "id:"):
            df = df.rename(columns={column: prefix+"_"+str(column)})
            i = i + 1
            print(i)
    return df