"""This module provides functions to print statistic.
    """

import numpy as np


def _print_vals_with_space(legend, number):
    print('{:<20}{:<20}'.format(legend+': ', number))


def print_stats(data):
    """This method prints the distribution of values inside the given data set.

    Prints: maximum, third quartile, mean, median, first quartile, minimum, standard deviation and variance

    :param data: The data to evaluate
    :type data: list(int)
    """
    _print_vals_with_space('Max', np.max(data))
    _print_vals_with_space('Third Quartile', np.percentile(data, 75))
    _print_vals_with_space('Mean', np.mean(data))
    _print_vals_with_space('Median', np.median(data))
    _print_vals_with_space('First Quartile', np.percentile(data, 25))
    _print_vals_with_space('Min', np.min(data))
    _print_vals_with_space('Standard deviation', np.std(data))
    _print_vals_with_space('Variance', np.var(data))


def print_frequency_stats_in_range(tag_data, start, end):
    """Prints an evaluation of the data in the given range.

    Prints number of tags in the range (absolute and percentage) and the cumulative usage (absolute and percentage)

    :param tag_data: The tag data to evaluate
    :type tag_data: list
    :param start: The first value of the range (exclusive)
    :type start: int
    :param end: The last value of the range (inclusive)
    :type end: int
    """
    tag_counts = [int(tag.attrib['Count']) for tag in tag_data]
    tags_in_range = [
        count for count in tag_counts if count > start and count <= end]
    tags_in_range_length = len(tags_in_range)
    tags_in_range_percentage = tags_in_range_length / len(tag_counts) * 100
    tags_in_range_cumulative_usage_percent = np.sum(
        tags_in_range) / np.sum(tag_counts) * 100
    print('{:<25}{:<30}{:<35.2f}{:<25}{:<30.2f}'.format('('+str(start)+', '+str(end)+']',
                                                        tags_in_range_length, tags_in_range_percentage, np.sum(tags_in_range), tags_in_range_cumulative_usage_percent))


def print_frequency_stats_in_ranges(tag_data, ranges):
    """Prints an evaluation of the data in the given ranges.

    Prints number of tags in the range (absolute and percentage) and the cumulative usage (absolute and percentage) for each range

    :param tag_data: The tag data to evaluate
    :type tag_data: list
    :param start: The first value of the range (exclusive)
    :type start: list(tuple(int))
    """
    print('{:<25}{:<30}{:<35}{:<25}{:<30}'.format('Usage frequency range', 'Number of tags in the range',
                                                  'Number of tags in the range [%]', 'Cumulative total usage', 'Cumulative total usage [%]'))
    for start, end in ranges:
        print_frequency_stats_in_range(tag_data, start, end)
