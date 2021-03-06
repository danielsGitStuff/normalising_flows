import multiprocessing
import threading
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import Process
from typing import List, Tuple


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
        self.pool = PoolReplacement(self.processes)
        result = self.pool.run_blocking(f, args)
        self.pool.close()
        self.pool = None
        if result is not None:
            return result()

    def join(self):
        combined_results = []
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
            combined_results.extend(results)
        return combined_results
