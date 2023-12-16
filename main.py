# -*- coding: utf-8 -*-
"""
@File : main.py
@Time : 2023/12/15 下午2:09
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
from common import EnvironmentalModel, policy_iteration, value_iteration
import numpy as np
import time

# Initialize environment
max_cars = 20
gamma = 0.9
theta = 0.1
em = EnvironmentalModel(lam_return1=3, lam_return2=2, lam_rental1=3, lam_rental2=4, max_cars=max_cars)

# Initialize value function and policy
policy = np.zeros((max_cars + 1, max_cars + 1), dtype=np.int16)
values = np.zeros((max_cars + 1, max_cars + 1))
states = [(s0, s1) for s0 in range(max_cars + 1) for s1 in range(max_cars + 1)]
# print(type(states[0]))

# Choose mode for policy iteration or value iteration
test1 = "policy_iteration"
test2 = "value_iteration"
mode = test1

# Start iteration
start_time = time.time()
if mode == test1:
    policy_iteration(states, values, policy, em, theta)
elif mode == test2:
    value_iteration(states, values, em, theta)
print("\nTime: {:.2f} seconds".format(time.time() - start_time))
