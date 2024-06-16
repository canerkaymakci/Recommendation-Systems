import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Reading and inspecting dataset.
main_df = pd.read_csv('Datasets/armut_data.csv')
df = main_df.copy()
df.head()
df.shape
df.isnull().sum()
df.info()

# Creating new column as "ServiceId_CategoryId" format.
df['Service'] = df['ServiceId'].astype(str) + '_' + df['CategoryId'].astype(str)

# Creating CartId as "UserId-Date"
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['NewDate'] = df.apply(lambda col: str(col['CreateDate'].year) + '-' + str(col['CreateDate'].month), axis=1)
df['Cart'] = df.apply(lambda col: str(col['UserId']) + '_' + col['NewDate'], axis=1)

# Creating every user' cart info.
services_df = df.groupby(['Cart', 'Service']).agg({'Service': 'count'}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Calculating every item set' support values.
frequent_itemsets = apriori(services_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values('support', ascending=False)

# Creating our association rules.
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)

# Test for item '2_0'
service_to_recommend = '2_0'
rules.sort_values('lift', ascending=False, inplace=True)
recommendation_list = []

for i, service in enumerate(rules['antecedents']):
    for j in list(service):
        if j == service_to_recommend:
            recommendation_list.append(list(rules.iloc[i]['consequents'])[0])

# Remove duplicates.
recommendation_list = list(dict.fromkeys(recommendation_list))


# Function for recommender.
def arl_recommender(rules, service_id, rec_count=-1, value='lift'):
    rules.sort_values(value, ascending=False, inplace=True)
    recommendation_list = []
    for i, service in enumerate(rules['antecedents']):
        for j in list(service):
            if j == service_id:
                recommendation_list.append(list(rules.iloc[i]['consequents'])[0])
    if len(recommendation_list) < rec_count:
        return recommendation_list[0:len(recommendation_list)]
    else:
        return recommendation_list[0:rec_count]


# Sample usages.
arl_recommender(rules, '2_0', 2)
arl_recommender(rules, '99_99', 2)
