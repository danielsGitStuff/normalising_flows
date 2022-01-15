from unittest import TestCase

from maf.variable.TrainingPlanner import TrainingPlanner
from maf.variable.VariableParam import LambdaParam


class TrainingPlannerTest(TestCase):
    def test_dependencies_1(self):
        # p = TrainingPlanner(LambdaParam('a', source_params=['b'], f=lambda b: 1),
        #                     LambdaParam('b', source_params=['c'], f=lambda c: 2),
        #                     LambdaParam('c', source_params=['a'], f=lambda: 3))
        p = TrainingPlanner(LambdaParam('a', source_params=['b', 'c'], f=lambda b, c: 1),
                            LambdaParam('b', source_params=['d'], f=lambda d: 2),
                            LambdaParam('c', source_params=['d'], f=lambda d: 3),
                            LambdaParam('d', source_params=['a'], f=lambda a: 4))
        p.build_plan()
