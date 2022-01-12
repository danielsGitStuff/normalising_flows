from typing import Dict, Set, List


class Dependency:
    def __init__(self, name: str, src: List[str]):
        self.name: str = name
        self.src: List[str] = src


class DependencyChecker:
    def __init__(self):
        self.dependencies_in: Dict[str, Set[str]] = {}
        self.dependencies_out: Dict[str, Set[str]] = {}

    def add_dependency(self, dep: Dependency):
        n = dep.name
        outs: Set[str] = self.dependencies_out.get(n, set())
        for s in dep.src:
            if s in outs:
                raise ValueError(f"Circular dependency between '{n}' and {sorted(outs)}")
        for s in dep.src:
            outs = self.dependencies_out.get(s, set())
            outs.add(n)
            for o in outs.copy():
                more_outs = self.dependencies_out.get(o, set())
                for more_o in more_outs:
                    outs.add(more_o)
                    more_ins = self.dependencies_in.get(more_o,set())
                    more_ins.add(s)
                    self.dependencies_in[more_o] = more_ins
                ins = self.dependencies_in.get(o, set())
                ins.add(s)
                self.dependencies_in[o] = ins
            self.dependencies_out[s] = outs

    def add_dependencies(self, dependencies: List[Dependency]):
        for d in dependencies:
            self.add_dependency(d)
            # print(f"state after adding '{d.name}'")
            # print(self.dependencies_in)
            # print(self.dependencies_out)
