
def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

def split_train_test(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items
    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df)*test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)
    else:
        total_user_list = create_user_list(df, user_size)
        print(len(total_user_list))
        train_user_list = [None] * len(total_user_list)
        test_user_list = [None] * len(total_user_list)
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list)*(1-test_size)):]
            train_item = item_list[:math.ceil(len(item_list)*(1-test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
    # Remove time
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    return train_user_list, test_user_list

def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair

