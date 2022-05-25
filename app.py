import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as pandas
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from apyori import apriori
import streamlit as st

# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")


# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]

        for item_pair in combinations(item_list, 2):
            yield item_pair


# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB',
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]


def function(data,min_support):


    data = data.fillna(0)

    data = data.astype('int')

    data.drop(0,inplace = True,axis = 1)

    data = data.applymap(str)

    records = []
    for i in range(data.shape[0]):
      records.append([str(data.values[i,j]) for j in range(0,data.shape[1])])

    te = TransactionEncoder()
    te_data = te.fit(records).transform(records)
    df = pd.DataFrame(te_data,columns=te.columns_)


    association_rules = apriori(df,min_support = min_support / 1000,min_length = 1 )
    association_results = list(association_rules)

    final_data = pd.DataFrame(association_results)

    final_data['length'] = final_data['items'].apply(lambda x: len(x))

    st.write(str(final_data['items']))


if __name__== "__main__":
    st.title("Webapp for Apriori Algorithm")
    main_bg = "background.jpg"
    main_bg_ext = "jpg"
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    new_title = '<p style="font-family:sans-serif; color:white; font-size: 20px;">@streamlit</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Apriori Web App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file = st.file_uploader(" Upload the  file here")


    val = st.text_input('Enter the min_support')
    # with open(file, 'r') as temp_f:
    #     # get No of columns in each line
    #     col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
    #
    # ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
    # column_names = [i for i in range(0, max(col_count))]
    #
    # ### Read csv
    # data = pd.read_csv(file, header=None, delimiter=",", names=column_names)
    if st.button("Generate"):
        with open(file.name, 'r') as temp_f:
            col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
        column_names = [i for i in range(0, max(col_count))]
        data = pd.read_csv(file, header=None, delimiter=",", names=column_names)
        function(data,int(val))

















