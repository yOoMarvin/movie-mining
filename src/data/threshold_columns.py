import pandas as pd

def thresoldByColumn (df, threshold, prefix):
    """
    Filters encoded columns.
    Takes only columns that have at least the threshold % ones
    :param df: oneHotEncoded DataFrame
    :param threshold: Percentage of ones in column
    :return: filtered DataFrame
    """
    size = len(df)
    filter_col = [col for col in df if col.startswith(prefix)]
    df_selected_columns = pd.DataFrame(df[filter_col])
    print(list(df_selected_columns))
    df.drop(filter_col, axis=1, inplace=True)
    columns = []
    for column in df_selected_columns:
        if not (df_selected_columns[column].value_counts()[1]/size >= threshold):
            columns.append(column)
    df_selected_columns.drop(columns, axis=1, inplace = True)
    df = pd.concat([df, df_selected_columns], axis=1)
    return df