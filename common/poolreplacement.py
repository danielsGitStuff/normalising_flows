import math
import sys
import threading
from concurrent.futures.process import ProcessPoolExecutor
from typing import List, Tuple
from multiprocessing import Process
import multiprocessing


class ThreadWrapper(threading.Thread):
    def __init__(self, f, args):
        super().__init__()
        self.f = f
        self.args = args

    def run(self) -> None:
        self.f(*self.args)


class PoolReplacement:
    def __init__(self, processes=None):
        if processes is None:
            processes = multiprocessing.cpu_count()

        # self.ts: List[ThreadWrapper] = []
        self.ts: List[Process] = []
        self.limit: int = processes
        self.params = []
        self.executor: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=processes)
        self.joined = False

    def apply_async(self, f, args):
        if self.joined:
            raise RuntimeError("pool has already been joined")
        self.params.append((f, args))

    def run_blocking(self, f, args):
        f = self.executor.submit(f, *args)
        return f.result()

    def join(self):
        if not self.joined:
            self.joined = True
            futures = []
            results = []
            for params in self.params:
                f = self.executor.submit(params[0], *params[1])
                futures.append(f)
            for f in futures:
                results.append(f.result())
            self.close()
            return results

    def close(self):
        self.executor.shutdown(wait=True)


class RestartingPoolReplacement:
    def __init__(self, processes: int = None):
        self.processes: int = processes
        self.pool: PoolReplacement = None
        self.argsList: List[List[any]] = []
        self.functionsList: List[any] = []

    def apply_async(self, f, args: Tuple[any, ...]):

        self.argsList.append(args)
        self.functionsList.append(f)

    def run_blocking(self, f, args: List[any]):
        pool = PoolReplacement(1)
        result = pool.run_blocking(f, args)
        pool.close()
        return result

    def join(self):
        combinedResults = []
        while len(self.argsList) > 0:
            amount_to_pop = self.processes
            if len(self.argsList) < amount_to_pop:
                amount_to_pop = len(self.argsList)
            self.pool = PoolReplacement(amount_to_pop)
            argsList = self.argsList[:amount_to_pop]
            functionsList = self.functionsList[:amount_to_pop]
            self.argsList = self.argsList[amount_to_pop:]
            self.functionsList = self.functionsList[amount_to_pop:]
            for f, args in zip(functionsList, argsList):
                self.pool.apply_async(f, args)
            results = self.pool.join()
            self.pool.close()
            self.pool = None
            combinedResults.extend(results)
        return combinedResults


if __name__ == '__main__':
    def f(s: str, i: int):
        for q in range(100000):
            print("{}.{}.{}".format(s, i, math.sqrt(q)))


    p = PoolReplacement()
    for i in range(10):
        p.apply_async(f, ("lala", i))
    p.join()
    print("END!")
    sys.exit(0)
