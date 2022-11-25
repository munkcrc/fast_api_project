import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple  # ,Any, Dict, Iterable,
from functools import cache  # ,cached_property
# from itertools import accumulate
# import pandas as pd


class CapCurve:

    def __init__(self,
                 data: List[Tuple[bool, float]]):
        self.data = data

    @property
    @cache
    def nr_of_true_outcomes(self) -> int:
        return sum([1 for t in self.data if t[0]])

    @property
    def nr_of_outcomes(self) -> int:
        return len(self.data)

    @property
    @cache
    def y_axis(self) -> List[float]:
        # sort data. When the first key (t[1]) returns that two elements are equal, the
        # second key (t[0]) is used to compare
        sorted_data = sorted(self.data, key=lambda t: (t[1], t[0]), reverse=True)
        sorted_outcomes = [int(t[0]) for t in sorted_data]
        y_axis_model = list(np.cumsum([0] + sorted_outcomes) / self.nr_of_true_outcomes)
        return y_axis_model

    def accuracy_ratio(self) -> float:
        # to calculate areas we use https://en.wikipedia.org/wiki/Trapezoidal_rule
        dx = 1 / self.nr_of_outcomes

        y_axis_perfect = list(
            np.arange(self.nr_of_true_outcomes + 1) / self.nr_of_true_outcomes) + [
                             1] * (self.nr_of_outcomes - self.nr_of_true_outcomes)

        # y_axis[0] = 0 and y_axis[-1] = 1 for both y_axis
        area_model = sum(self.y_axis[1:-1]) * dx + 0.5 * dx - 0.5
        area_perfect = sum(y_axis_perfect[1:-1]) * dx + 0.5 * dx - 0.5
        return area_model/area_perfect

    def roc_auc(self) -> float:
        # AR = 2*AUC - 1
        return (self.accuracy_ratio() + 1)/2

    def plot_curve(
            self,
            plot_perfect_model: bool = True,
            title: str = '',
            ar_in_title: bool = True,
    ) -> None:
        x_axis = [i / self.nr_of_outcomes for i in range(self.nr_of_outcomes + 1)]
        cr_blue = '#003749'
        cr_pink = '#ff6a70'
        plt.figure()
        lw = 2
        plt.plot(x_axis, self.y_axis, color=cr_pink,  # '#FF9999',  # 'coral',
                 lw=lw, label='This model')
        plt.plot(
            [0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='Uniform Model')

        if plot_perfect_model:
            # y_axis_perfect = [
            #     i / self.nr_of_true_outcomes
            #     for i in range(self.nr_of_true_outcomes + 1)] + [
            #     1]*(self.nr_of_outcomes-self.nr_of_true_outcomes)

            y_axis_perfect = list(
                np.arange(self.nr_of_true_outcomes + 1) / self.nr_of_true_outcomes) + [
                                 1] * (self.nr_of_outcomes - self.nr_of_true_outcomes)

            plt.plot(x_axis, y_axis_perfect, color=cr_blue,  # '#16365C',  # 'navy',
                     lw=lw, label='Perfect model')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if title:
            title = f' {title},'
        if ar_in_title:
            plt.title(f'CAP Curve,{title} AR = {self.accuracy_ratio():.3f}')
        else:
            plt.title('CAP Curve')
        plt.xlabel("Percentage of total sample")
        plt.ylabel("Percentage of positive outcomes")
        plt.legend(loc="lower right")
        plt.show()
