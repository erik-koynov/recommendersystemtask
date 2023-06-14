import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator

BAR_PLOT_FIG_HEIGHT_COEFF = 16
BAR_PLOT_FIG_WIDTH_COEFF = 30


class IndexMissmatchError(Exception):
    pass

class GroupedBarPlot:
    """
    Wrapper around the matplotlib bar plot functionality to create a grouped bar plot of variable number of arrays.
    """
    def __init__(self,
                *value_arrays: Tuple[pd.Series],
                bar_width: float,
                title: str,
                legend_labels: List[str] = None,):


        check_indices(*value_arrays)
        self.value_arrays = value_arrays
        self.bar_width = bar_width

        if legend_labels is None:
            legend_labels = [None] * len(value_arrays)
        self.legend_labels = legend_labels
        self.x_tick_positions = np.arange(len(self.value_arrays[0].index))
        self.title = title
        self.fig: plt.Figure = plt.figure(figsize=(BAR_PLOT_FIG_WIDTH_COEFF * self.bar_width * int(len(self.value_arrays)*2/2),
                                              BAR_PLOT_FIG_HEIGHT_COEFF * self.bar_width * int(len(self.value_arrays)*1.5/2)))
        self.ax: plt.Axes = self.fig.add_subplot(111)

    def show(self, *value_arrays_for_annotations: Tuple[pd.Series]):
        return self._grouped_bar_plot(*value_arrays_for_annotations).show()

    def _grouped_bar_plot(self, *value_arrays_for_annotations: Tuple[pd.Series]) -> plt.Figure:

        if len(value_arrays_for_annotations) == 0:
            value_arrays_for_annotations = self.value_arrays

        for i, bar_container in enumerate(self.group_basic_bar_plots_generator()):
            for j, bar in enumerate(bar_container):

                plt.annotate(f"{value_arrays_for_annotations[i].iloc[j]}", (bar.get_x(), bar.get_height()), fontsize=9)

        _ = self.ax.set_xticks(self.x_tick_positions, labels=self.value_arrays[0].index, rotation=90, fontsize=15)

        _ = self.ax.legend(bbox_to_anchor=(0.7, 0.5, 0.5, 0.5))
        _ = self.ax.set_title(self.title)

        return self.fig

    def group_basic_bar_plots_generator(self):
        return group_basic_bar_plots_generator(*self.value_arrays,
                                               x_tick_positions = self.x_tick_positions,
                                               bar_width = self.bar_width,
                                               axis = self.ax,
                                               legend_labels= self.legend_labels)
def group_basic_bar_plots_generator(*value_arrays,
                                    x_tick_positions: np.ndarray,
                                    bar_width: float,
                                    axis: plt.Axes,
                                    legend_labels: List[str]) -> Generator[BarContainer, None, None]:
    """
    Return a generator of containers, which are correctly ordered to build a group bar plot.
    :param value_arrays:
    :param x_tick_positions:
    :param bar_width:
    :param axis:
    :param legend_labels:
    :return:
    """
    n_arrays = len(value_arrays)
    mid_point: float = int(n_arrays / 2.)
    is_even = (n_arrays % 2 == 0)

    for i, values in enumerate(value_arrays):
        is_left_of_middle = (i < mid_point)
        increment_direction = -1 if is_left_of_middle else 1

        if not is_even:
            increment_coeff = abs(mid_point - i) * 2
        else:
            increment_coeff =  abs(mid_point - i) * 2 + increment_direction

        adjusted_tick_positions = (x_tick_positions + increment_coeff * increment_direction * (bar_width/2))
        yield axis.bar(adjusted_tick_positions,
                       height=values, width=bar_width,
                       label=legend_labels[i])


def check_indices(*value_arrays: pd.Series) -> None:
    for i in range(len(value_arrays)-1):
        if len(value_arrays[i].index.symmetric_difference(value_arrays[i+1].index)):
            raise IndexMissmatchError(f"The value_arrays must have the same index values,"
                                      f" but {value_arrays[i].index, value_arrays[i+1].index} differ.")



