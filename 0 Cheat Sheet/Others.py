#  Convert List to DataFrame
X = pd.DataFrame(data=X)


#  Add Column to DataFrame
df['new_col'] = mylist
#Alternative - Convert the list to a series or array and then assign:
se = pd.Series(mylist)
df['new_col'] = se.values
#or
df['new_col'] = np.array(mylist)
#  Let df, be your dataset, and mylist the list with the values you want to add to the dataframe.
column_values = pd.Series(mylist)
#Then use the insert function to add the column.
# This function lets you choose in which position you want to place the column by setting loc=0
df.insert(loc=0, column='new_column', value=column_values)

#  Export your pandas DataFrame to a CSV file:
df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv')

#  Rename DataFrame Columns.
X.columns = ['A','B','C','D','E','F','G','H','I']
