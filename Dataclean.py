import pandas as pd
from itertools import product

users = pd.read_csv('user.csv',engine='python'
)
actions = pd.read_csv(
    'action.csv',engine='python'
)
cats = pd.read_csv(
    'category.csv',engine='python'
)

#feature engineering
cats.shoplevel = pd.Categorical(cats.shoplevel)
cats['shoplevel']=cats.shoplevel.cat.codes
print(cats['shoplevel'])

users.gender = pd.Categorical(users.gender)
users["gender"] = users.gender.cat.codes

users.age_range = pd.Categorical(users.age_range)
users["age_range"] = users.age_range.cat.codes
users.occupation = pd.Categorical(users.occupation)
users["occupation"] = users.occupation.cat.codes

users.location = pd.Categorical(users.location)
users["location"] = users.location.cat.codes
cats["cat_id"] = cats["cat_id"].astype(str)
users["user_id"] = users["user_id"].astype(str)
actions["cat_id"] = actions["cat_id"].astype(str)
actions["user_id"] = actions["user_id"].astype(str)
actions['action_type']=product(actions['action_type'],actions['age_range'])

#behavior sequence generation
actions_group = actions.sort_values(by=["timestamp"]).groupby("user_id")
actions_data = pd.DataFrame(
    data={
        "user_id": list(actions_group.groups.keys()),
        "cat_id": list(actions_group.cat_id.apply(list)),
        "action_type": list(actions_group.action_type.apply(list)),
        "timestamp": list(actions_group.timestamp.apply(list)),
    }
)
pd.set_option('display.max_columns', None)

sequence_length = 6
step = 1

def create_sequences(values, w_size, step):
    sequences = []
    initial_index = 0
    while True:
        end_index = initial_index + w_size
        seq = values[initial_index:end_index]
        if len(seq) < w_size:
            seq = values[-w_size:]
            if len(seq) == w_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        initial_index += step
    return sequences

actions_data.cat_id= actions_data.cat_id.apply(
    lambda ids: create_sequences(ids, sequence_length, step)
)

actions_data.action_type = actions_data.action_type.apply(
    lambda ids: create_sequences(ids, sequence_length, step)
)


actions_data_cat = actions_data[["user_id", "cat_id"]].explode(
    "cat_id", ignore_index=True
)
actions_data_action_type = actions_data[["action_type"]].explode("action_type", ignore_index=True)

actions_data_transformed = pd.concat([actions_data_cat, actions_data_action_type], axis=1)

actions_data_transformed = actions_data_transformed.join(
    users.set_index("user_id"), on="user_id"
)


actions_data_transformed.cat_id= actions_data_transformed.cat_id.apply(
    lambda x: ",".join(x)
)
actions_data_transformed.action_type = actions_data_transformed.action_type.apply(
    lambda x: ",".join([str(v) for v in x])
)

#train_data,test_data,