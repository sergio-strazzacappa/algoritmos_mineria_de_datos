import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

MIN_SUPPORT = 2 / 9

dataset = [
    ["I1", "I2", "I5"],
    ["I2", "I4"],
    ["I2", "I3"],
    ["I1", "I2", "I4"],
    ["I1", "I3"],
    ["I2", "I3"],
    ["I1", "I3"],
    ["I1", "I2", "I3", "I5"],
    ["I1", "I2", "I3"]
]

# Encode the dataset into Panda's Dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df, "\n")

# Generate the frequent itemsets using the apriori algorithm
frequent_itemsets = apriori(
    df, min_support=MIN_SUPPORT, use_colnames=True, verbose=True)
print(frequent_itemsets)

# Generate the association rules
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=0.9)
print(rules)