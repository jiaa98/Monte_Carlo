# encoding = 'utf-8'
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings


def feature_values_count(dataframe, feature_name):
    """
    :param dataframe:
    :param feature_name:
    :return: Analyze the visualization of different features
    """
    dataframe[feature_name].value_counts().sort_index().plot(kind='barh')
    sns.despine()
    plt.show()


def neighbor_Sale_price(dataframe, Neighborhood='Neighborhood', SalePrice='SalePrice'):
    """
    :param dataframe:
    :param Neighborhood:
    :param SalePrice:
    :return: Check the average price of houses sold in different locations, mainly to see that those locations are better, sorted in descending order
    """
    dataframe.groupby(Neighborhood)[SalePrice].mean().sort_values().plot(kind='barh', figsize=(9, 12))
    sns.despine()
    plt.show()


def garage_centralAir_neighbor_Sale_price(dataframe,
                                          GarageCars='GarageCars',
                                          CentralAir='CentralAir',
                                          Neighborhood='Neighborhood',
                                          SalePrice='SalePrice'):
    """
    :param dataframe:
    :param GarageCars:
    :param CentralAir:
    :param Neighborhood:
    :param SalePrice:
    :return: The visualized garage contains the average housing price when the car is accommodated, whether there is central air-conditioning, the location of the lot, etc.
    """
    dataframe.groupby([GarageCars, CentralAir, Neighborhood])[SalePrice].mean().sort_values().plot(kind='barh', figsize=(9, 12))
    sns.despine()
    plt.show()


def monte_Carlo(dataframe, GarageCars_list, CentralAir_list, Neighborhood_list, max_level, total_money):
    """
    :param dataframe:
    :param GarageCars_list:
    :param CentralAir_list:
    :param Neighborhood_list:
    :param max_level:
    :param total_money:
    :return: Monte Carlo simulation,
         1. When the total price is low, the garage houses a large number of cars and houses with central air-conditioning. Generally, the probability of a house with a poor location is higher.
         2. When the total price is low, the garage can not accommodate the number of cars and does not contain central air-conditioning houses. Generally, the probability of the house location is relatively better.

    """

    L = []
    count = 0
    while count < max_level:
        GarageCars = random.choice(GarageCars_list)
        CentralAir = random.choice(CentralAir_list)
        Neighborhood = random.choice(Neighborhood_list)
        d1 = dataframe[dataframe['GarageCars'] == GarageCars]
        d2 = d1[d1['CentralAir'] == CentralAir]
        d3 = d2[d2['Neighborhood'] == Neighborhood]
        if len(d3) == 0:
            continue
        minSalePrice = min(list(d3['SalePrice']))
        if minSalePrice < total_money:
            L.append([f"GarageCars:{GarageCars},CentralAir:{CentralAir},Neighborhood:{Neighborhood}"])

        count += 1
        print(f"count:{count}")

    dict_all = {}
    for item in L:
        if item[0] not in dict_all:
            dict_all[item[0]] = 0
        else:
            dict_all[item[0]] += 1

    sort_dict = sorted(dict_all.items(), key=lambda x: x[1], reverse=True)

    return sort_dict


if __name__ == "__main__":

    GarageCars_list_ = [0, 1, 2, 3, 4]
    CentralAir_list_ = ['N', 'Y']
    Neighborhood_list_ = \
        ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert',
         'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
         'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']

    """Read data"""
    data = pd.read_csv("./train.csv", header=0, encoding='utf-8')

    """Check the average price of houses with car capacity in the garage"""
    feature_values_count(data, feature_name='GarageCars')

    """Check the average price of houses with central air-conditioning"""
    feature_values_count(data, feature_name='CentralAir')

    """Check the price of houses in different locations"""
    neighbor_Sale_price(data)

    """Check the number of cars that the garage can accommodate, whether it contains central air-conditioning, and the price of the house in different locations, etc."""
    garage_centralAir_neighbor_Sale_price(data)

    """Using Monte Carlo simulation to verify, suppose we only have 130,000 assets in hand, whether the choice of simulating a house to buy a house meets the assumptions"""
    res = monte_Carlo(dataframe=data,
                      GarageCars_list=GarageCars_list_,
                      CentralAir_list=CentralAir_list_,
                      Neighborhood_list=Neighborhood_list_,
                      max_level=10000,
                      total_money=130000
                      )

    """Print the selected results of the simulation to prove the correctness of the hypothesis"""
    print(res)

    #A total of 1000 simulations, the number of GarageCars contained in the garage is 2, whether it contains Central Air-N does not contain it, the location of the house (Neighborhood)-oldTown, the Monte Carlo random simulation selection strategy hits 126 times
