# -*- coding: utf-8 -*-
"""
@File : common.py
@Time : 2023/12/15 上午10:42
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""

# renting
# returning

import numpy as np
import math
# from scipy.stats import poisson
import matplotlib.pyplot as plt


def draw_fig(
        value: np.ndarray,
        policy: np.ndarray,
        iteration: int = 0,
        mode: str = "None",
        max_car_num: int = 20,
        max_move_num: int = 5) -> None:
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    # ax.matshow(policy, cmap=plt.cm.plasma, vmin=-max_move_num, vmax=max_move_num)
    # ax.matshow(policy, cmap=plt.cm.bwr, vmin=-max_move_num, vmax=max_move_num)
    ax.matshow(policy, cmap=plt.cm.RdYlGn, vmin=-max_move_num, vmax=max_move_num)
    ax.set_xticks(range(max_car_num + 1))
    ax.set_yticks(range(max_car_num + 1))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    for x in range(max_car_num + 1):
        for y in range(max_car_num + 1):
            ax.text(x=x, y=y, s=int(policy.T[x, y]), va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_{}$'.format(iteration), fontsize=20)

    y, x = np.meshgrid(range(max_car_num + 1), range(max_car_num + 1))
    ax = fig.add_subplot(122, projection='3d')
    # ax.scatter3D(y, x, value.T,c="darkslategray")
    # ax.scatter3D(y, x, value.T)
    ax.scatter3D(y, x, value.T,c="goldenrod")
    ax.set_xlim3d(0, max_car_num)
    ax.set_ylim3d(0, max_car_num)
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    ax.set_title('value for ' + r'$\pi_{}$'.format(iteration), fontsize=20)
    plt.savefig(f'./{mode}/{iteration}.png', bbox_inches='tight')


def poisson_mod(lam: int, k: int) -> np.ndarray:
    """
    Return poisson PMF clipped at max_k with remaining tail probability
    placed at max_k.
    """
    pmf = np.zeros(k + 1)
    for i in range(k):
        pmf[i] = math.exp(-lam) * (lam ** i) / math.factorial(i)
    # pmf = poisson.pmf(np.arange(k + 1), lam)
    pmf[k] = 1 - np.sum(pmf)

    return pmf


def build_rent_return_pmf(lam_rent: int, lam_return: int, max_cars: int = 20) -> np.ndarray:
    """
    Return p(new_rentals, returns | initial_cars) as numpy array:
        p[initial_cars, new_rentals, returns]
    """
    pmf = np.zeros((max_cars + 1, max_cars + 1, max_cars + 1))

    for init_cars in range(max_cars + 1):
        new_rentals_pmf = poisson_mod(lam_rent, init_cars)
        for new_rentals in range(init_cars + 1):
            max_returns = max_cars - init_cars + new_rentals
            returns_pmf = poisson_mod(lam_return, max_returns)
            for returns in range(max_returns + 1):
                prob = returns_pmf[returns] * new_rentals_pmf[new_rentals]
                pmf[init_cars, new_rentals, returns] = prob

    return pmf


class EnvironmentalModel(object):
    """Environment model of Jack's Car Rental"""

    def __init__(self,
                 lam_return1: int,
                 lam_return2: int,
                 lam_rental1: int,
                 lam_rental2: int,
                 max_cars: int = 20) -> None:
        # pre-build the rentals/returns pmf for each location
        self.rent_return_pmf = []
        self.rent_return_pmf.append(build_rent_return_pmf(lam_rental1, lam_return1, max_cars))
        self.rent_return_pmf.append(build_rent_return_pmf(lam_rental2, lam_return2, max_cars))
        self.max_cars = max_cars

    def get_transition_model(self, s: tuple, a: int) -> (dict, dict):
        """
        Return 2-tuple:
            1. p(s'| s, a) as dictionary:
                keys = s' values = p(s' | s, a)
            2. E(r | s, a, s') as dictionary:
                keys = s' values = E(r | s, a, s')
        """
        s = (s[0] - a, s[1] + a)  # move a cars from loc1 to loc2
        move_cost = -math.fabs(a) * 2  # ($2) per car moved
        # move_cost = -action * 2  # ($2) per car moved
        t_prob = [{}, {}]
        expected_r = [{}, {}]
        for loc in {0, 1}:
            morning_cars = s[loc]
            rent_return_pmf = self.rent_return_pmf[loc]
            for rents in range(morning_cars + 1):
                max_returns = self.max_cars - morning_cars + rents
                for returns in range(max_returns + 1):
                    prob = rent_return_pmf[morning_cars, rents, returns]
                    if prob < 1e-5:
                        continue
                    s_prime = morning_cars - rents + returns
                    r = rents * 10
                    t_prob[loc][s_prime] = t_prob[loc].get(s_prime, 0) + prob
                    expected_r[loc][s_prime] = expected_r[loc].get(s_prime, 0) + prob * r

        # join probabilities and expectations from loc1 and loc2
        t_model = {}
        r_model = {}
        for s_prime1 in t_prob[0]:
            p1 = t_prob[0][s_prime1]
            for s_prime2 in t_prob[1]:
                p2 = t_prob[1][s_prime2]
                t_model[(s_prime1, s_prime2)] = p1 * p2
                # expectation of reward calculated using p(s', r | s, a)
                # need to normalize by p(s' | s, a) to get E(r | s, a, s')
                norm_E1 = expected_r[0][s_prime1] / p1
                norm_E2 = expected_r[1][s_prime2] / p2
                r_model[(s_prime1, s_prime2)] = norm_E1 + norm_E2 + move_cost

        return t_model, r_model


def policy_iteration(
        states: list,
        values: np.ndarray,
        policy: np.ndarray,
        em: EnvironmentalModel,
        theta: float = 0.1,
        gamma: float = 0.9,
        max_cars: int = 20) -> None:
    # Policy Iteration (using iterative policy evaluation)
    iteration = 0
    while (True):
        i = 0
        print('\nPolicy at iteration = {}:'.format(iteration))
        # print(policy)
        print('\nPolicy evaluation iteration = {}. v(s) delta:'.format(iteration))
        # Policy Evaluation
        while (True):
            delta = 0
            for state in states:
                value = values[state]
                t_model, r_model = em.get_transition_model(state, policy[state])
                value_new = 0
                for state_prime in t_model:
                    p, r = t_model[state_prime], r_model[state_prime]
                    value_new += p * (gamma * values[state_prime] + r)
                values[state] = value_new
                delta = max(delta, abs(value_new - value))
            print('Iteration {}: max delta = {:.2f}'.format(i, delta))
            i += 1
            if delta < theta:
                break

        # Policy Iteration
        stable = True
        for state in states:
            old_action = policy[state]
            value_best = -100
            max_a = min(5, state[0], max_cars - state[1])
            min_a = max(-5, -state[1], -(max_cars - state[0]))
            for a in range(min_a, max_a + 1):
                value = 0
                (t_model, r_model) = em.get_transition_model(state, a)
                for state_prime in t_model:
                    p, r = t_model[state_prime], r_model[state_prime]
                    value += p * (gamma * values[state_prime] + r)
                if value > value_best:
                    policy[state] = a
                    value_best = value
            if policy[state] != old_action:
                stable = False
        draw_fig(values, policy, iteration, "policy")
        iteration += 1
        if stable:
            return


def value_iteration(states: list,
                    values: np.ndarray,
                    policy: np.ndarray,
                    em: EnvironmentalModel,
                    theta: float = 0.1,
                    gamma: float = 0.9,
                    max_cars: int = 20) -> None:
    # Value Iteration
    value_best = -100
    # theta = 0.5  # value(s) delta stopping threshold
    print('Worst |value_old(s) - value(s)| delta:')
    iteration = 0
    while (True):
        delta = 0
        values_old = values.copy()
        values = np.zeros((max_cars + 1, max_cars + 1))
        for state in states:
            value_best = -100
            max_a = min(5, state[0], max_cars - state[1])
            min_a = max(-5, -state[1], -(max_cars - state[0]))
            for a in range(min_a, max_a + 1):
                t_model, r_model = em.get_transition_model(state, a)
                value_new = 0
                for state_prime in t_model:
                    p = t_model[state_prime]
                    r = r_model[state_prime]
                    # must use previous iteration's values(state): value_olds(state)
                    value_new += p * (gamma * values_old[state_prime] + r)
                if value_new > value_best:
                    policy[state] = a
                    value_best = value_new
                value_best = max(value_best, value_new)
            values[state] = value_best
            delta = max(delta, abs(values[state] - values_old[state]))
        print('Iteration {}: max delta = {:.2f}'.format(iteration, delta))
        iteration += 1
        if delta < theta: break

    print('\nValue iteration done, final policy:')
    print(policy)
    draw_fig(values, policy, mode="value")
