from typing import List

import tensorflow as tf

from distributions.LearnedDistribution import FitHistory, EarlyStop


class LearnedDistributionPlaceholder:
    def __init__(self):
        self.transformed_distribution = ["tfd placeholder"]


class LearnedDistributionTestTest(tf.test.TestCase):
    def setUp(self):
        self.t1s = [6, 7, 1, 2, 3, 4]
        self.history: FitHistory = FitHistory()
        self.results: List[bool] = []

    def test_stop_2(self):
        es = EarlyStop("t1", tf.greater_equal, 2)
        es.learned_distribution = LearnedDistributionPlaceholder()
        for epoch, t1 in enumerate(self.t1s):
            self.history.add("t1", t1)
            try:
                b = es.on_epoch_end(epoch, self.history)
                print(b)
                self.results.append(b)
            except RuntimeError:
                assert len(self.results) == 5
                assert self.results == [False, False, False, False, True]

    def test_stop_1(self):
        es = EarlyStop("t1", tf.greater_equal, 1)
        es.learned_distribution = LearnedDistributionPlaceholder()
        for epoch, t1 in enumerate(self.t1s):
            self.history.add("t1", t1)
            try:
                b = es.on_epoch_end(epoch, self.history)
                print(b)
                self.results.append(b)
            except RuntimeError:
                assert len(self.results) == 4
                assert self.results == [False, False, False, True]

    def test_end(self):
        es = EarlyStop("t1", tf.greater_equal, 10)
        es.learned_distribution = LearnedDistributionPlaceholder()
        t1s = [5, 6, 1, 2, 3]
        for epoch, t1 in enumerate(t1s):
            self.history.add("t1", t1)
            try:
                es.learned_distribution = LearnedDistributionPlaceholder()
                if t1 == 6:
                    es.learned_distribution.transformed_distribution = ["123"]
                b = es.on_epoch_end(epoch, self.history)
                print(b)
                self.results.append(b)
            except RuntimeError:
                assert len(self.results) == 4
                assert self.results == [False, False, False, True]
        assert len(self.results) == 5
        assert self.results == [False, False, False, False, False]
        assert es.learned_distribution.transformed_distribution == LearnedDistributionPlaceholder().transformed_distribution
        es.after_training_ends()
        assert es.learned_distribution.transformed_distribution == ["123"]