def get_col_indices(df, col_names):
    column_indices = [df.columns.get_loc(col) for col in col_names]
    return column_indices