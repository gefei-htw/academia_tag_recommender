import numpy as np


def print_val(legend, number):
    print('{:<20}{:<20}'.format(legend+': ', number))


def print_stats(data):
    print_val('Max', np.max(data))
    print_val('Third Quartile', np.percentile(data, 75))
    print_val('Mean', np.mean(data))
    print_val('Median', np.median(data))
    print_val('First Quartile', np.percentile(data, 25))
    print_val('Min', np.min(data))
    print_val('Standard deviation', np.std(data))
    print_val('Variance', np.var(data))


def print_frequency_stats(sorted_tags, start, end):
    amount_of_tags = len(sorted_tags)
    overall_tag_usage = np.sum(
        list(map(lambda x: int(x.attrib['Count']), sorted_tags)))
    tags_in_range = list(filter(lambda x: int(x.attrib['Count']) > start and int(
        x.attrib['Count']) <= end, sorted_tags))
    range_length = len(tags_in_range)
    range_percentage = range_length / amount_of_tags * 100
    range_cumulative_usage = 0
    for tag in tags_in_range:
        range_cumulative_usage += int(tag.attrib['Count'])
    range_cumulative_usage_percent = range_cumulative_usage / overall_tag_usage * 100
    print('{:<25}{:<30}{:<35.2f}{:<25}{:<30.2f}'.format('('+str(start)+', '+str(end)+']',
                                                        range_length, range_percentage, range_cumulative_usage, range_cumulative_usage_percent))
