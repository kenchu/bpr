import pandas as pd

from bpr import BPR
import bpr1

pd_train = pd.read_csv("data/train.txt", sep="\t")
pd_test = pd.read_csv("data/test.txt", sep="\t")
pd_valid = pd.read_csv("data/vali.txt", sep="\t")
# print(pd_train.describe())


# df, user_mapping = convert_unique_idx(df, 'user')
# df, item_mapping = convert_unique_idx(df, 'item')
# train_user_list, test_user_list = split_train_test(df,
#                                                      user_size,
#                                                      test_size=args.test_size,
#                                                      time_order=args.time_order)
# bpr = BPR.forward()

user_idx = bpr1.convert_unique_idx(pd_train, "user_id")
item_idx = bpr1.convert_unique_idx(pd_train, "item_id")

train_pair = bpr1.create_pair(user_idx)
print(train_pair)