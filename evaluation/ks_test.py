import numpy as np
from scipy import stats


def ks_test(data_1, data_2, disp=False):
    # p_value_list = []
    # statistic_list = []
    result = dict()
    for col in data_1.columns:
        res = stats.ks_2samp(data_1[col], data_2[col])
        result[col] = res[1]
        # p_value_list.append(res[1])
        # statistic_list.append(res[0])
    # n = np.argmax(p_value_list)
    # p_value = p_value_list[n]
    # statistic = statistic_list[n]
    # if disp:
    #     print('Column     p_value         statistic')
    #     for idx, col in enumerate(data_1.columns):
    #         print(col, p_value_list[idx], statistic_list[idx])
    # return [p_value, statistic, data_1.columns[n], p_value_list, statistic_list]
    return result
