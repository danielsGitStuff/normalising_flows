from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

from setproctitle import setproctitle
from typing import Callable, Union, Dict, Any, Tuple, List, Optional

from normalising_flows.src.common import NotProvided
from normalising_flows.src.common.jsonloader import Ser, SerSettings
from normalising_flows.src.distributions import enable_memory_growth

ProzessArgs = Optional[Union[Dict[str, Any], Tuple, List]]
import time


class ProzessTestUtils:
    @staticmethod
    def static_wait(seconds: int, msg_when_done: str = 'done sleeping'):
        setproctitle(f"asd waiting {seconds}s")
        time.sleep(seconds)
        print(f"'{msg_when_done}' ,{seconds}s")
        return 'asd'


class WorkLoad(Ser):
    @staticmethod
    def static_method_call(method: Callable, args: ProzessArgs) -> Any:
        if args is None:
            return method()
        if isinstance(args, Dict):
            return method(**args)
        return method(*args)

    @staticmethod
    def static_workload_call(work: WorkLoad) -> Any:
        return work.run()

    @staticmethod
    def create_static_workload(method: Callable = NotProvided, args: ProzessArgs = NotProvided, use_tf: bool = False) -> StaticMethodWorkload:
        return StaticMethodWorkload(method=method, args=args, use_tf=use_tf)

    def __init__(self, use_tf: bool = False):
        super().__init__()
        self.use_tf: bool = use_tf
        self.debug_result = None

    def run(self):
        if self.use_tf:
            enable_memory_growth()
        return self._run()

    def _run(self):
        raise NotImplementedError()

    def debug(self) -> WorkLoad:
        """execute directly"""
        for _ in range(3):
            print('Workload.debug() called', file=sys.stderr)
        self.debug_result = self.run()
        return self


class StaticMethodWorkload(WorkLoad):
    def __init__(self, method: Callable = NotProvided, args: ProzessArgs = NotProvided, use_tf: bool = False):
        super().__init__(use_tf=use_tf)
        self.method: Callable = method
        self.args: ProzessArgs = args

    def _run(self) -> Any:
        return WorkLoad.static_method_call(self.method, self.args)


class BatchedWorkload(WorkLoad):
    def __init__(self, work: List[WorkLoad] = NotProvided, use_tf: bool = NotProvided):
        super().__init__(use_tf=use_tf)
        self.work: List[WorkLoad] = work

    def run(self):
        if self.use_tf:
            enable_memory_growth()
        for w in self.work:
            w._run()


class Prozess(Ser):
    @staticmethod
    def static_execute(work: WorkLoad) -> Any:
        s = Prozess(work)
        return s.run()

    def __init__(self, work: WorkLoad = NotProvided):
        super().__init__()
        self.sub_pool: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=1)
        self.work: WorkLoad = work
        self.ignored.add('sub_pool')

    def run(self) -> Any:
        setproctitle('asd single')
        f = self.sub_pool.submit(WorkLoad.static_workload_call, work=self.work)
        self.sub_pool.shutdown()
        r = f.result()
        return r


class Prozessor:
    def __init__(self, max_workers: int = 1):
        self.max_workers: int = max_workers
        self.master_pool: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.work: List[WorkLoad] = []
        self.closed: bool = False

    def run_later(self, work: WorkLoad) -> Prozessor:
        self.work.append(work)
        return self

    def close(self):
        self.master_pool.shutdown(wait=False, cancel_futures=True)

    def join(self) -> List:
        if self.closed:
            raise RuntimeError('pool already closed!')
        self.closed = True
        futures = []
        for w in self.work:
            f = self.master_pool.submit(Prozess.static_execute, work=w)
            futures.append(f)
        self.master_pool.shutdown()
        results = [f.result() for f in futures]
        return results


if __name__ == '__main__':
    SerSettings.enable_testing_mode()
    setproctitle('asd main')
    w1 = StaticMethodWorkload(ProzessTestUtils.static_wait, args=(4,))
    w2 = StaticMethodWorkload(ProzessTestUtils.static_wait, args=(2,))
    pz = Prozessor()
    pz.run_later(w1).run_later(w2)
    pz.join()
    print('exit')
