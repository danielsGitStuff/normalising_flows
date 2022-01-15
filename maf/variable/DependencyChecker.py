from typing import Dict, Set, List


class CircularDependencyError(BaseException):
    pass


class MissingDependencyError(BaseException):
    pass


class Dependency:
    def __init__(self, name: str, src: List[str] = []):
        self.name: str = name
        self.src: List[str] = src


class DependencyChecker:
    def __init__(self):
        self.dependencies_in: Dict[str, Set[str]] = {}
        self.dependencies_out: Dict[str, Set[str]] = {}

    def circle_check(self, dep: Dependency):
        n = dep.name
        outs: Set[str] = self.dependencies_out.get(n, set())
        for s in dep.src:
            if s in outs:
                raise CircularDependencyError(f"Circular dependency between '{n}' and {sorted(outs)}")
        for s in dep.src:
            outs = self.dependencies_out.get(s, set())
            outs.add(n)
            for o in outs.copy():
                more_outs = self.dependencies_out.get(o, set())
                for more_o in more_outs:
                    outs.add(more_o)
                    more_ins = self.dependencies_in.get(more_o, set())
                    more_ins.add(s)
                    self.dependencies_in[more_o] = more_ins
                ins = self.dependencies_in.get(o, set())
                ins.add(s)
                self.dependencies_in[o] = ins
            self.dependencies_out[s] = outs

    def check_dependencies(self, dependencies: List[Dependency]):
        available: Set[str] = {d.name for d in dependencies}
        required: Set[str] = set()
        [required.add(s) for d in dependencies for s in d.src]
        missing: Set[str] = set()
        [missing.add(r) for r in required if r not in available]
        if len(missing) > 0:
            raise MissingDependencyError(f"missing dependencies: {sorted(missing)}")
        for d in dependencies:
            self.circle_check(d)
