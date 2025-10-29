import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

min_sup = 0.2
min_conf = 0.9

DATASET_NUMBER = 7
dataset = []

if DATASET_NUMBER == 1:
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
elif DATASET_NUMBER == 2:
    dataset = [
        ["Yerba", "Azucar"],
        ["Chips", "Obleas", "Cacao"],
        ["Chips", "Mermelada", "Cacao"],
        ["Yerba", "Mermelada", "Leche"],
        ["Yerba", "Mermelada", "Leche"],
        ["Yerba", "Azucar", "Obleas", "Cacao"]
    ]
    min_sup = 1 / 3
    min_conf = 0.6
elif DATASET_NUMBER == 3:
    dataset = [
        ["A", "B", "C", "D", "E", "F"],
        ["A", "D", "F", "I", "J"],
        ["B", "D", "E", "K"],
        ["B", "C", "D", "E", "F", "G"],
        ["A", "D", "E", "H"]
    ]
    min_sup = 0.6
    min_conf = 0.8
elif DATASET_NUMBER == 4:
    dataset = [
        ["A", "B", "E"],
        ["B", "D"],
        ["B", "C"],
        ["A", "B", "D"],
        ["A", "C", "F"],
        ["B", "C"],
        ["A", "C"],
        ["A", "B", "C", "E"],
        ["A", "B", "C"]
    ]
    min_sup = 2 / 9
    min_conf = 0.8
elif DATASET_NUMBER == 5:
    dataset = [
        ["A", "B", "D"],
        ["B", "C", "E"],
        ["A", "B", "C", "E"],
        ["B", "E"],
        ["A", "B", "C", "E"],
    ]
    min_sup = 0.6
    min_conf = 0
elif DATASET_NUMBER == 6:
    dataset = [
        ["L", "P", "C"],
        ["L", "Z", "P"],
        ["P", "M", "C"],
        ["L", "P", "C", "Z"],
        ["Z", "P"],
        ["L", "P", "M"]
    ]
    min_sup = 0.5
    min_conf = 0
elif DATASET_NUMBER == 7:
    dataset = [
        ["L", "P", "C"],
        ["L", "Z"],
        ["P", "C"],
        ["L", "P", "C", "Z"],
        ["Z"],
        ["L", "P"]
    ]
    min_sup = 0.5
    min_conf = 0

# Encode the dataset into Panda's Dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df, "\n")

# Generate the frequent itemsets using the fp-growth algorithm
frequent_itemsets = fpgrowth(
    df, min_support=min_sup, use_colnames=True, verbose=True)

print(frequent_itemsets, "\n")

# Generate the association rules
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_conf)

print(rules[[
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence"
    ]])
