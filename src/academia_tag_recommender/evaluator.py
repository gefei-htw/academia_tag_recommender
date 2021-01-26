"""This module handles evaluations."""
from sklearn.metrics import hamming_loss as hamming_loss_score, accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    """This evaluator calculates different measures for for given data.
    """

    def __init__(self, y, prediction):
        self.hamming_loss = hamming_loss_score(y, prediction)
        self.accuracy = accuracy_score(y, prediction)

        self.precision_samples = precision_score(
            y, prediction, average='samples', zero_division=0)
        self.precision_macro = precision_score(
            y, prediction, average='macro', zero_division=0)
        self.precision_micro = precision_score(
            y, prediction, average='micro', zero_division=0)

        self.recall_samples = recall_score(
            y, prediction, average='samples', zero_division=0)
        self.recall_macro = recall_score(
            y, prediction, average='macro', zero_division=0)
        self.recall_micro = recall_score(
            y, prediction, average='micro', zero_division=0)

        self.f1_samples = f1_score(
            y, prediction, average='samples', zero_division=0)
        self.f1_macro = f1_score(
            y, prediction, average='macro', zero_division=0)
        self.f1_micro = f1_score(
            y, prediction, average='micro', zero_division=0)

    def print_stats(self):
        """Print hamming loss, precision, recall and f1 as example-based measures and micro and macro results for precision, recall and f1."""
        print('{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}'.format('', 'Hamming Loss', 'Accuracy',
                                                                'Precision', 'Recall', 'F1 '))
        print('{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}'.format('samples', self.hamming_loss,
                                                            self.accuracy, self.precision_samples, self.recall_samples, self.f1_samples))
        print('{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}'.format(
            'micro', '', '', self.precision_micro, self.recall_micro, self.f1_micro))
        print('{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}'.format(
            'macro', '', '', self.precision_macro, self.recall_macro, self.f1_macro))

    def __str__(self):
        return ''
