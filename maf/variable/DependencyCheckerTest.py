from unittest import TestCase

from maf.variable.DependencyChecker import Dependency, DependencyChecker


class DependencyCheckerTest(TestCase):
    def test_circle_1(self):
        checker = DependencyChecker()
        try:
            checker.add_dependencies([Dependency('a', src=['b', 'c']),
                                      Dependency('b', src=['d']),
                                      Dependency('c', src=['d']),
                                      Dependency('d', src=['a'])])
        except ValueError as e:
            print('success, circular dependency detected')
            return
        self.assertTrue(False)
