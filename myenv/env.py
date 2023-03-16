import logging

logging.basicConfig()
logger = logging.getLogger("LOGGING_SCIMAI-Gym_V1")
logger.setLevel(logging.WARN)

import numpy as np

from datetime import datetime
from itertools import chain
# from tabulate import tabulate
from timeit import default_timer
# from IPython.display import display

# import collections
# import dataframe_image as dfi
# import glob
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
import gym
from gym.spaces import Box
# import os
# import pandas as pd
# import random
# import seaborn as sns
# import shutil

class State:
    """
    We choose the state vector to include all current stock levels for each 
    warehouse and product type, plus the last demand values.
    """

    def __init__(self, product_types_num, distr_warehouses_num, T,
                 demand_history, t=0):
        self.product_types_num = product_types_num
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype=np.int32)
        self.distr_warehouses_num = distr_warehouses_num
        self.distr_warehouses_stocks = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)
        self.T = T
        self.demand_history = demand_history
        self.t = t

        logger.debug(f"\n--- State --- __init__"
                     f"\nproduct_types_num is "
                     f"{self.product_types_num}"
                     f"\nfactory_stocks is "
                     f"{self.factory_stocks}"
                     f"\ndistr_warehouses_num is "
                     f"{self.distr_warehouses_num}"
                     f"\ndistr_warehouses_stocks is "
                     f"{self.distr_warehouses_stocks}"
                     f"\nT is "
                     f"{self.T}"
                     f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

    def to_array(self):
        logger.debug(f"\n--- State --- to_array"
                     f"\nnp.concatenate is "
                     f"""{np.concatenate((
                         self.factory_stocks,
                         self.distr_warehouses_stocks.flatten(),
                         np.hstack(list(chain(*chain(*self.demand_history)))),
                         [self.t]))}""")

        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten(),
            np.hstack(list(chain(*chain(*self.demand_history)))),
            [self.t]))

    def stock_levels(self):
        logger.debug(f"\n--- State --- stock_levels"
                     f"\nnp.concatenate is "
                     f"""{np.concatenate((
                         self.factory_stocks,
                         self.distr_warehouses_stocks.flatten()))}""")

        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten()))
    
class Action:
    """
    The action vector consists of production and shipping controls.
    """

    def __init__(self, product_types_num, distr_warehouses_num):
        self.production_level = np.zeros(
            (product_types_num,),
            dtype=np.int32)
        self.shipped_stocks = np.zeros(
            (distr_warehouses_num, product_types_num),
            dtype=np.int32)

        logger.debug(f"\n--- Action --- __init__"
                     f"\nproduction_level is "
                     f"{self.production_level}"
                     f"\nshipped_stocks is "
                     f"{self.shipped_stocks}")
        
class SupplyChainEnvironment:
    """
    We designed a divergent two-echelon supply chain that includes a single 
    factory, multiple distribution warehouses, and multiple product types over 
    a fixed number of time steps. At each time step, the agent is asked to find 
    the number of products to be produced and preserved at the factory, as well 
    as the number of products to be shipped to different distribution 
    warehouses. To make the supply chain more realistic, we set capacity 
    constraints on warehouses (and consequently, on how many units to produce 
    at the factory), along with storage and transportation costs. 
    """

    def __init__(self):
        # number of product types (e.g., 2 product types)
        self.product_types_num = 2
        # number of distribution warehouses (e.g., 2 distribution warehouses)
        self.distr_warehouses_num = 2
        # final time step (e.g., an episode takes 25 time steps)
        self.T = 25

        # maximum demand value, units (e.g., [3, 6])
        self.d_max = np.array(
            [3, 6],
            np.int32)
        # maximum demand variation according to a uniform distribution,
        # units (e.g., [2, 1])
        self.d_var = np.array(
            [2, 1],
            np.int32)

        # sale prices, per unit (e.g., [20, 10])
        self.sale_prices = np.array(
            [20, 10],
            np.int32)
        # production costs, per unit (e.g., [2, 1])
        self.production_costs = np.array(
            [2, 1],
            np.int32)

        # storage capacities for each product type at each warehouse,
        # units (e.g., [[3, 4], [6, 8], [9, 12]])
        self.storage_capacities = np.array(
            [[3, 4], [6, 8], [9, 12]],
            np.int32)

        # storage costs of each product type at each warehouse,
        # per unit (e.g., [[6, 3], [4, 2], [2, 1]])
        self.storage_costs = np.array(
            [[6, 3], [4, 2], [2, 1]],
            np.float32)
        # transportation costs of each product type for each distribution
        # warehouse, per unit (e.g., [[.1, .3], [.2, .6]])
        self.transportation_costs = np.array(
            [[.1, .3], [.2, .6]],
            np.float32)

        # penalty costs, per unit (e.g., [10, 5])
        self.penalty_costs = .5*self.sale_prices

        print(f"\n--- SupplyChainEnvironment --- __init__"
              f"\nproduct_types_num is "
              f"{self.product_types_num}"
              f"\ndistr_warehouses_num is "
              f"{self.distr_warehouses_num}"
              f"\nT is "
              f"{self.T}"
              f"\nd_max is "
              f"{self.d_max}"
              f"\nd_var is "
              f"{self.d_var}"
              f"\nsale_prices is "
              f"{self.sale_prices}"
              f"\nproduction_costs is "
              f"{self.production_costs}"
              f"\nstorage_capacities is "
              f"{self.storage_capacities}"
              f"\nstorage_costs is "
              f"{self.storage_costs}"
              f"\ntransportation_costs is "
              f"{self.transportation_costs}"
              f"\npenalty_costs is "
              f"{self.penalty_costs}")

        self.reset()

    def reset(self, demand_history_len=5):
        # (five) demand values observed
        self.demand_history = collections.deque(maxlen=demand_history_len)

        logger.debug(f"\n--- SupplyChainEnvironment --- reset"
                     f"\ndemand_history is "
                     f"{self.demand_history}")

        for d in range(demand_history_len):
            self.demand_history.append(np.zeros(
                (self.distr_warehouses_num, self.product_types_num),
                dtype=np.int32))
        self.t = 0

        logger.debug(f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

    def demand(self, j, i, t):
        # we simulate a seasonal behavior by representing the demand as a
        # co-sinusoidal function with a stochastic component (a random variable
        # assumed to be distributed according to a uniform distribution),
        # in order to evaluate the agent
        demand = self.d_max[i-1]/2 + self.d_max[i-1]/2*np.cos(4*np.pi*(2*j*i+t)/self.T) + np.random.randint(0, self.d_var[i-1]+1)

        logger.debug(f"\n--- SupplyChainEnvironment --- demand"
                     f"\nj is "
                     f"{j}"
                     f"\ni is "
                     f"{i}"
                     f"\nt is "
                     f"{t}"
                     f"\ndemand is "
                     f"{demand}")

        return demand

    def initial_state(self):
        logger.debug(f"\n--- SupplyChainEnvironment --- initial_state"
                     f"\nState is "
                     f"""{State(
                         self.product_types_num, self.distr_warehouses_num, 
                         self.T, list(self.demand_history))}""")

        return State(self.product_types_num, self.distr_warehouses_num,
                     self.T, list(self.demand_history))

    def step(self, state, action):
        demands = np.fromfunction(
            lambda j, i: self.demand(j+1, i+1, self.t),
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)

        logger.debug(f"\n--- SupplyChainEnvironment --- step"
                     f"\nstate is "
                     f"{state}"
                     f"\nstate.factory_stocks is "
                     f"{state.factory_stocks}"
                     f"\nstate.distr_warehouses_stocks is "
                     f"{state.distr_warehouses_stocks}"
                     f"\naction is "
                     f"{action}"
                     f"\naction.production_level is "
                     f"{action.production_level}"
                     f"\naction.shipped_stocks is "
                     f"{action.shipped_stocks}"
                     f"\ndemands is "
                     f"{demands}")

        # next state
        next_state = State(self.product_types_num, self.distr_warehouses_num,
                           self.T, list(self.demand_history))

        next_state.factory_stocks = np.minimum(
            np.subtract(np.add(state.factory_stocks,
                               action.production_level),
                        np.sum(action.shipped_stocks, axis=0)
                        ),
            self.storage_capacities[0]
        )

        for j in range(self.distr_warehouses_num):
            next_state.distr_warehouses_stocks[j] = np.minimum(
                np.subtract(np.add(state.distr_warehouses_stocks[j],
                                   action.shipped_stocks[j]),
                            demands[j]
                            ),
                self.storage_capacities[j+1]
            )

        logger.debug(f"\n-- SupplyChainEnvironment -- next state"
                     f"\nnext_state is "
                     f"{next_state}"
                     f"\nnext_state.factory_stocks is "
                     f"{next_state.factory_stocks}"
                     f"\nnext_state.distr_warehouses_stocks is "
                     f"{next_state.distr_warehouses_stocks}"
                     f"\nnext_state.demand_history is "
                     f"{next_state.demand_history}"
                     f"\nnext_state.t is "
                     f"{next_state.t}")

        # revenues
        total_revenues = np.dot(self.sale_prices,
                                np.sum(demands, axis=0))
        # production costs
        total_production_costs = np.dot(self.production_costs,
                                        action.production_level)
        # transportation costs
        total_transportation_costs = np.dot(
            self.transportation_costs.flatten(),
            action.shipped_stocks.flatten())
        # storage costs
        total_storage_costs = np.dot(
            self.storage_costs.flatten(),
            np.maximum(next_state.stock_levels(),
                       np.zeros(
                           ((self.distr_warehouses_num+1) *
                            self.product_types_num),
                           dtype=np.int32)
                       )
        )
        # penalty costs (minus sign because stock levels would be already
        # negative in case of unfulfilled demand)
        total_penalty_costs = -np.dot(
            self.penalty_costs,
            np.add(
                np.sum(
                    np.minimum(next_state.distr_warehouses_stocks,
                               np.zeros(
                                   (self.distr_warehouses_num,
                                    self.product_types_num),
                                   dtype=np.int32)
                               ),
                    axis=0),
                np.minimum(next_state.factory_stocks,
                           np.zeros(
                               (self.product_types_num,),
                               dtype=np.int32)
                           )
            )
        )
        # reward function
        reward = total_revenues - total_production_costs - \
            total_transportation_costs - total_storage_costs - \
            total_penalty_costs

        logger.debug(f"\n-- SupplyChainEnvironment -- reward"
                     f"\ntotal_revenues is "
                     f"{total_revenues}"
                     f"\ntotal_production_costs is "
                     f"{total_production_costs}"
                     f"\ntotal_transportation_costs is "
                     f"{total_transportation_costs}"
                     f"\ntotal_storage_costs is "
                     f"{total_storage_costs}"
                     f"\ntotal_penalty_costs is "
                     f"{total_penalty_costs}"
                     f"\nreward is "
                     f"{reward}")

        # the actual demand for the current time step will not be known until
        # the next time step. This implementation choice ensures that the agent
        # may benefit from learning the demand pattern so as to integrate a
        # sort of demand forecasting directly into the policy
        self.demand_history.append(demands)
        # actual time step value is not observed (for now)
        self.t += 1

        logger.debug(f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

        logger.debug(f"\n-- SupplyChainEnvironment -- return"
                     f"\nnext_state is "
                     f"{next_state}, "
                     f"\nreward is "
                     f"{reward}, "
                     f"\ndone is "
                     f"{self.t == self.T-1}")

        return next_state, reward, self.t == self.T-1
    
class SupplyChain(gym.Env):
    """
    Gym environment wrapper.
    """

    def __init__(self, config):
        self.reset()

        # low values for action space (no negative actions)
        low_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1) *
             self.supply_chain.product_types_num),
            dtype=np.int32)
        # high values for action space
        high_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1) *
             self.supply_chain.product_types_num),
            dtype=np.int32)
        # high values for action space (factory)
        high_act[
            :self.supply_chain.product_types_num
        ] = np.sum(self.supply_chain.storage_capacities, axis=0)
        # high values for action space (distribution warehouses, according to
        # storage capacities)
        high_act[
            self.supply_chain.product_types_num:
        ] = (self.supply_chain.storage_capacities.flatten()[
            self.supply_chain.product_types_num:])
        # action space
        self.action_space = Box(low=low_act,
                                high=high_act,
                                dtype=np.int32)

        # low values for observation space
        low_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype=np.int32)
        # low values for observation space (factory, worst case scenario in
        # case of non-production and maximum demand)
        low_obs[
            :self.supply_chain.product_types_num
        ] = -np.sum(self.supply_chain.storage_capacities[1:], axis=0) * \
            self.supply_chain.T
        # low values for observation space (distribution warehouses, worst case
        # scenario in case of non-shipments and maximum demand)
        low_obs[
            self.supply_chain.product_types_num:
                (self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num
        ] = np.array([
            -(self.supply_chain.d_max+self.supply_chain.d_var) *
            self.supply_chain.T
        ] * self.supply_chain.distr_warehouses_num).flatten()
        # high values for observation space
        high_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype=np.int32)
        # high values for observation space (factory and distribution
        # warehouses, according to storage capacities)
        high_obs[
            :(self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num
        ] = self.supply_chain.storage_capacities.flatten()
        # high values for observation space (demand, according to the maximum
        # demand value)
        high_obs[
            (self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num:
            len(high_obs)-1
        ] = np.array([
            self.supply_chain.d_max+self.supply_chain.d_var] *
            len(list(chain(*self.supply_chain.demand_history)))).flatten()
        # high values for observation space (episode, according to the final
        # time step)
        high_obs[len(high_obs)-1] = self.supply_chain.T
        # observation space
        self.observation_space = Box(low=low_obs,
                                     high=high_obs,
                                     dtype=np.int32)

        logger.debug(f"\n--- SupplyChain --- __init__"
                     f"\nlow_act is "
                     f"{low_act}"
                     f"\nhigh_act is "
                     f"{high_act}"
                     f"\naction_space is "
                     f"{self.action_space}"
                     f"\nlow_obs is "
                     f"{low_obs}"
                     f"\nhigh_obs is "
                     f"{high_obs}"
                     f"\nobservation_space is "
                     f"{self.observation_space}")

    def reset(self):
        self.supply_chain = SupplyChainEnvironment()
        self.state = self.supply_chain.initial_state()

        logger.debug(f"\n--- SupplyChain --- reset"
                     f"\nsupply_chain is "
                     f"{self.supply_chain}"
                     f"\nstate is "
                     f"{self.state}"
                     f"\nstate.to_array is "
                     f"{self.state.to_array()}")

        return self.state.to_array()

    def step(self, action):
        # casting to integer actions (units of product to produce and ship)
        action_obj = Action(
            self.supply_chain.product_types_num,
            self.supply_chain.distr_warehouses_num)
        action_obj.production_level = action[
            :self.supply_chain.product_types_num].astype(np.int32)
        action_obj.shipped_stocks = action[
            self.supply_chain.product_types_num:
        ].reshape((self.supply_chain.distr_warehouses_num,
                   self.supply_chain.product_types_num)).astype(np.int32)

        logger.debug(f"\n--- SupplyChain --- step"
                     f"\naction is "
                     f"{action}"
                     f"\naction_obj is "
                     f"{action_obj}"
                     f"\naction_obj.production_level is "
                     f"{action_obj.production_level}"
                     f"\naction_obj.shipped_stocks is "
                     f"{action_obj.shipped_stocks}")

        self.state, reward, done = self.supply_chain.step(
            self.state, action_obj)

        logger.debug(f"\n-- SupplyChain -- return"
                     f"\nstate.to_array is "
                     f"{self.state.to_array()}"
                     f"\nreward is "
                     f"{reward}"
                     f"\ndone is "
                     f"{done}")

        return self.state.to_array(), reward, done, {}