from unittest import TestCase

from maf.variable.DependencyChecker import Dependency, DependencyChecker, CircularDependencyError, MissingDependencyError


class DependencyCheckerTest(TestCase):
    def test_circle_1(self):
        checker = DependencyChecker()
        try:
            checker.check_dependencies([Dependency('a', src=['b', 'c']),
                                        Dependency('b', src=['d']),
                                        Dependency('c', src=['d']),
                                        Dependency('d', src=['a'])])
        except CircularDependencyError as e:
            print(e)
            print('success, circular dependency detected')
            return
        self.assertTrue(False)

    def test_missing_1(self):
        checker = DependencyChecker()
        try:
            checker.check_dependencies([Dependency('a', src=['b', 'c']),
                                        Dependency('b', src=['d']),
                                        Dependency('c', src=['d']),
                                        Dependency('d', src=['e'])])
        except MissingDependencyError as e:
            print(e)
            print('success, missing dependency detected')
            return
        self.assertTrue(False)
