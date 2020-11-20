import numpy as np

def printVal(legend, number):
    print('{:<20}{:<20}'.format(legend+': ', number))

def printStats(data):
    printVal('Max', np.max(data))
    printVal('Third Quartile', np.percentile(data, 75))
    printVal('Mean', np.mean(data))
    printVal('Median', np.median(data))
    printVal('First Quartile', np.percentile(data, 25))
    printVal('Min', np.min(data))
    printVal('Standard deviation', np.std(data))
    printVal('Variance', np.var(data))
    
def printFrequencyStats(sortedTags, start, end):
    amountOfTags = len(sortedTags)
    overallTagUsage = np.sum(list(map(lambda x: int(x.attrib['Count']), sortedTags)))
    tagsInRange = list(filter(lambda x: int(x.attrib['Count']) > start and int(x.attrib['Count']) <= end, sortedTags))
    rangeLength = len(tagsInRange)
    rangePercentage = rangeLength / amountOfTags * 100
    rangeCumulativeUsage = 0
    for tag in tagsInRange: 
        rangeCumulativeUsage += int(tag.attrib['Count'])
    rangeCumulativeUsagePercent = rangeCumulativeUsage / overallTagUsage * 100
    print('{:<25}{:<30}{:<35.2f}{:<25}{:<30.2f}'.format('('+str(start)+', '+str(end)+']', rangeLength, rangePercentage, rangeCumulativeUsage, rangeCumulativeUsagePercent))