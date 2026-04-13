import abc
import logging
import multiprocessing
import multiprocessing.process
import os
import queue
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import SimpleQueue
from multiprocessing.managers import BaseManager
from queue import Queue
from typing import Any, Callable, Iterable, List, Literal, Optional, Union

from packaging import version

from .errors import NoAcceleratorFoundError, TaskError
from .log import BlockingQueueHandler, nolog


class TaskState(abc.ABC):
    """Empty class to save state of a task"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def shouldstop(self) -> bool:
        """Check whether worker should stop or not.
        It is to support collaborative terminate functionality.
        """
        pass

    @abc.abstractmethod
    def setstop(self):
        """Set stop event to give worker a hint for stopping"""
        pass


class ThreadState(TaskState):
    """
    Object of this class will be passed
    worker threads to keep their state
    """

    def __init__(self, shutdown: threading.Event):
        super().__init__()
        self._shutdown = shutdown

    def shouldstop(self) -> bool:
        return self._shutdown.is_set()

    def setstop(self):
        """Set stop event for thread to give it
        a hint for stopping"""
        self._shutdown.set()


class ProcessState(TaskState):
    """
    Object of this class will be passed
    worker processes to keep their state
    """

    def __init__(self, shutdown: multiprocessing.Event):
        super().__init__()
        self._shutdown = shutdown

    def shouldstop(self) -> bool:
        """Check whether worker thread should stop or not.
        It is to support collaborative terminate functionality
        for thread
        """
        return self._shutdown.is_set()

    def setstop(self):
        """Set stop event for thread to give it
        a hint for stopping"""
        self._shutdown.set()


@dataclass(frozen=True)
class _AcceleratorStat:
    """
    Immutable record of a single accelerator device's basic statistics.

    Attributes
    ----------
    type_ : str
        The type of accelerator, e.g. "cuda" for NVIDIA GPUs, "tpu" for Google TPUs,
        "MPS" for Apple Metal Performance Shaders, etc.
    deviceid : str
        A string identifier for the device. For GPUs this would be "0", "1"
    freemem : int
        Amount of free memory available on the device, expressed in bytes.
    totalmem : int
        Amount of total memory available on the device, expressed in bytes.
    """

    type_: str
    deviceid: str
    freemem: int
    totalmem: int


@dataclass(frozen=True)
class _RunArgs(object):
    ppid: int
    mtctx: str
    rank: int
    loglevel: int
    state: TaskState
    shutdownevt: Union[threading.Event, multiprocessing.Event]
    sendack: bool
    iq: Union[Queue, SimpleQueue]
    oq: Union[Queue, SimpleQueue]
    init: Optional[Callable[[int, logging.Logger, TaskState, ...], Any]]
    initargs: tuple[Any]
    func: Callable[[tuple[Any]], Any]


class ParallelExecutorBase(abc.ABC):
    """Base class for parallel executors e.g.
    ProcessParallel and ThreadParallel"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def apply(self, func: Callable[[Any], Any], it: Iterable[Any]) -> Iterable[Any]:
        pass


def _get_cuda_stats_torch() -> List[_AcceleratorStat]:
    import torch

    installed = version.parse(torch.__version__)
    # 1.9.0 (June 2021) is minimum version where
    # torch.cuda.mem_get_info was introduced
    required = version.parse("1.9.0")
    if installed < required:
        raise RuntimeError(
            f"PyTorch {required} or higher is required, but {installed} is installed."
        )

    ret = []
    ndevices = torch.cuda.device_count()
    for device in range(ndevices):
        try:
            # TODO: test with pytorch 1.9.0
            with torch.cuda.device(device):
                free, total = torch.cuda.mem_get_info()
        except RuntimeError:
            # Covers torch.cuda.AcceleratorError as well
            continue
        ret.append(_AcceleratorStat("cuda", str(device), free, total))
    return ret


def gputasks(
    tasksize: int,
    # TODO: add support for tf, numba, cupy
    backend: Literal["torch"],
    include: Optional[List[int]] = None,
    exclude: Optional[List[int]] = None,
    maxtaskpergpu: int = -1,
    maxtasks: int = -1,
    niceness: float = 0.0,
) -> List[int]:
    """
    Return list of GPU IDs that workers can split among themselves.

    Parameters
    ----------
    include : list[int], optional
        GPUs explicitly allowed.
    exclude : list[int], optional
        GPUs to skip. Both exclude and include cannot
        be given simultaenously.
    tasksize : int
        Task size in bytes. Used to check if GPU has enough free memory.
    maxtaskspergpu : int, default=-1
        Maximum number of tasks allowed per GPU. By default as many as
        possible.
    maxtasks : int, default=-1
        Global cap on total tasks across all GPUs. By default as many as
        possible.
    niceness : float, default=0.0
        Minimum fraction of memory to leave

    Returns
    -------
    list[int]
        GPU IDs assigned to the worker.
    """
    if include is not None and exclude is not None:
        raise TypeError("Both :include list and :exclude list cannot be given")

    if tasksize < 1024:
        raise ValueError(":tasksize cannot be less than 1K")

    if backend == "torch":
        candidates = _get_cuda_stats_torch()
    else:
        raise ValueError(f"Unsupported backend {backend!r}")

    if not candidates:
        raise NoAcceleratorFoundError(
            f"No GPUs found. Please check you {backend!r} installation"
        )

    if include is not None:
        candidates = [
            candidate for candidate in candidates if int(candidate.deviceid) in include
        ]
    if exclude is not None:
        candidates = [
            candidate
            for candidate in candidates
            if int(candidate.deviceid) not in exclude
        ]

    if not candidates:
        # There are no candidates left after inclusion or exclusion
        return []

    available = []
    for candidate in candidates:
        toleave = candidate.totalmem * niceness
        freemem = candidate.freemem - toleave
        if freemem <= 0.0:
            # Device usage is alreay above threshold
            continue

        nworkers = int(freemem / tasksize)
        if maxtaskpergpu != -1 and nworkers > maxtaskpergpu:
            nworkers = maxtaskpergpu

        for _ in range(nworkers):
            available.append(int(candidate.deviceid))
            if maxtasks != -1 and len(available) >= maxtasks:
                return available
    return available


class LocalParallel(ParallelExecutorBase):
    """
    Run tasks concurrently using multiple processes or threads with controlled buffering,
    safe logging, and activity monitoring.

    This class provides functionality similar to `multiprocessing.Pool` and
    `concurrent.futures.ProcessPoolExecutor`, but with several key differences:

    1. **Bounded Buffers**
       Input (`inpbuffsize`) and output (`outbuffsize`) queues are size limited
       to reduce memory pressure and handle cases where producer and consumer
       speeds differ significantly.

    2. **Multi Thread and Multi processing Safe Logging**
       A lightweight logger is wrapped in a multiprocessing-safe handler so
       that worker processes can log without risk of interleaved or corrupted
       output.

    3. **Process Rank Awareness**
       Each worker is assigned a process number (rank), enabling scenarios
       where workers must bind to specific hardware resources (e.g., GPUs).

    4. **Inactivity Monitoring**
       Both workers and the producer are monitored for inactivity via
       `inactivitytimeout`. Warnings are issued when input or output stalls,
       helping detect bottlenecks or deadlocks early.

    Parameters
    ----------
    njobs : int
        Number of worker processes to launch. Each worker runs tasks independently.
        Example: `njobs=4` will use 4 processes concurrently.
    init : Callable[[int, logging.Logger, TaskState, ...], Any], optional
        Optional initialization function called once per worker before processing
        begins. Receives the worker's rank, multiprocessing-safe logger and task state object
        for saving initialized objects. Useful for setting up resources such as
        model initialization and database connections.
    initargs : tuple[Any], optional
        Extra arguments passed to the `init` function. Allows customization of
        worker setup beyond rank and logger.
    inpbuffsize : int, default=64
        Maximum number of items buffered in the input queue. Lower values reduce
        memory usage but may slow throughput if producers are fast.
    outbuffsize : int, default=64
        Maximum number of results buffered in the output queue. Lower values help
        control memory pressure when consumers are slower than producers.
    logger : logging.Logger, default=nolog()
        Logger instance used by workers. Automatically wrapped to avoid interleaved
        log lines across processes. Default logger logs are not printed.
    mtctx : multi tasking context, optional
        Multi tasking context (`thread`, `fork`, `spawn`, or `forkserver`). If not provided,
        `thread` context is used.
    inactivitytimeout : float, default=0.0
        Maximum allowed inactivity (in seconds) for workers or producer before a
        warning is logged. Set to >0.0 to detect stalls in input or output flow.

    Example
    -------
    >>> from utilitybarn.log import stderr
    >>> from utilitybarn.task import ProcessParallel, gputasks
    >>>
    >>> def init(rank, logger, state, *args):
    >>>     state.rank = rank
    >>>     state.logger = logger
    >>>     gpus = args[0]
    >>>     logger.info(f"Worker with rank {rank} Using GPU {gpus[rank]}")
    >>>
    >>>
    >>> def mul(state, args):
    >>>     a, b = args
    >>>     state.logger.info(f"Multiplying {a} and {b}")
    >>>     return a * b
    >>>
    >>>
    >>> tasks = gputasks(tasksize=2048, maxtasks=4, backend="torch")
    >>> p = ProcessParallel(
    >>>     njobs=len(tasks), init=init, initargs=(tasks,), logger=stderr()
    >>> )
    >>> for res in p.apply(mul, [(20, 30), (40, 50)]):
    >>>     print(res)
    >>>

    Limitations
    -----------
    1. Workers that get killed due to external signals
       or crashes are not restarted and their inactivity
       time is not updated
    2. No proper support for sharing tensors across
       processes.
    3. Same queue is used for logging and output which
       can cause contention if workers are producing
       logs at higher rate
    """

    _RCVD_MESSAGE = "R"
    _OUT_MESSAGE = "O"

    def __init__(
        self,
        *,
        njobs: int,
        init: Optional[Callable[[int, logging.Logger, TaskState, ...], Any]] = None,
        initargs: Optional[tuple[Any]] = None,
        inpbuffsize: int = 64,
        outbuffsize: int = 64,
        logger: logging.Logger = nolog(),
        mtctx: Literal["thread", "fork", "spawn", "forkserver"] = "thread",
        inactivitytimeout: float = 0.0,
    ):
        super().__init__()
        if njobs <= 0:
            raise ValueError(f":njobs cannot be <1 but got {njobs}")
        self._njobs = njobs
        self._init = init
        self._initargs = initargs or ()
        self._inpbuffsize = inpbuffsize
        self._outbuffsize = outbuffsize
        self._logger = logger
        self._mtctx = mtctx
        if mtctx == "thread":
            self._mpctx = None
        elif mtctx in ("fork", "spawn", "forkserver"):
            self._mpctx = multiprocessing.get_context(mtctx)
        else:
            raise ValueError(f"Unknown :mtctx {mtctx!r}")
        self._inactivitytimeout = inactivitytimeout
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._feedshutdown = threading.Event()
        # Mutable state
        self._running = False
        self._workers = None
        self._states = None
        self._activity = None
        self._lastsent = None
        self._inprunning = False

    def apply(self, func: Callable[[Any], Any], it: Iterable[Any]) -> Iterable[Any]:
        """Run a `func` in parallel across multiple processes or threads based on `mtctx`
        and yield results as they become available.

        This method is designed to distribute work items from an iterable to
        `njobs` workers and collect their outputs.
        Results are yielded as soon as workers finish processing,
        which means the order of outputs is **not retained** relative to the input
        sequence. It is upto programmer how they want to identify result e.g. returning
        some id from `func` as part of its return value.

        Parameters
        ----------
        func : Callable[[Any], Any]
            The function to apply to each item. Must be picklable and safe to run
            in a separate process.
        it : Iterable[Any]
            The input iterable providing items to process.

        Returns
        -------
        Iterable[Any]
            A generator yielding results from worker processes as they complete.

        Example
        -------
        >>> from utilitbarn.task import LocalParallel
        >>> p = LocalParallel(njobs=4)
        >>> results = p.apply(lambda _, x: x * x, range(10))
        >>> for r in results:
        >>>     print(r)
        # Output will contain squares of 0-9, but not necessarily in order.
        """

        self._ensure_pid()
        with self._lock:
            if self._running:
                raise RuntimeError("Already running")
            self._running = True
        self._feedshutdown.clear()

        manager = None
        kbint = False
        try:
            if self._mpctx is not None:
                manager = self._mpctx.Manager()
            yield from self._apply(func, it, manager)
        except KeyboardInterrupt as ex:
            kbint = True
            raise KeyboardInterrupt(str(ex)) from None
        finally:
            self._feedshutdown.set()
            if self._mtctx == "thread":
                self._stop_threads(self._workers, kbint=kbint)
            else:
                self._stop_processes(self._workers, manager)
            self._workers = None
            self._states = None
            self._activity = None
            self._lastsent = None
            self._inprunning = False
            with self._lock:
                self._running = False

    def _apply(
        self,
        func: Callable[Any, Any],
        it: Iterable[Any],
        manager: Optional[BaseManager],
    ) -> Iterable[Any]:
        if manager is not None:
            # It is multi processing
            iq = manager.Queue(self._inpbuffsize)
            oq = manager.Queue(self._outbuffsize)
        else:
            # It is multi threading
            iq = Queue(self._inpbuffsize)
            oq = Queue(self._outbuffsize)

        loglevel = self._logger.level if self._logger.hasHandlers() else None
        self._workers = []
        self._activity = OrderedDict()
        for rank in range(self._njobs):
            if self._mpctx is not None:
                shutdownevt = self._mpctx.Event()
                state = ProcessState(shutdownevt)
            else:
                shutdownevt = threading.Event()
                state = ThreadState(shutdownevt)
            shutdownevt.clear()
            args = _RunArgs(
                ppid=self._pid,
                mtctx=self._mtctx,
                rank=rank,
                loglevel=loglevel,
                state=state,
                shutdownevt=shutdownevt,
                sendack=self._inactivitytimeout != 0.0,
                iq=iq,
                oq=oq,
                init=self._init,
                initargs=self._initargs,
                func=func,
            )
            try:
                if self._mpctx is not None:
                    worker = self._mpctx.Process(
                        target=LocalParallel._run,
                        name=None,
                        args=(args,),
                        daemon=True,
                    )
                    worker.start()
                    # Ensure PID is actually assigned by the OS
                    while worker.pid is None:
                        time.sleep(0.001)
                    id_ = worker.pid
                else:
                    worker = threading.Thread(
                        group=None,
                        name=None,
                        target=LocalParallel._run,
                        args=(args,),
                        daemon=True,
                    )
                    worker.start()
                    id_ = worker.ident
            except (multiprocessing.ProcessError, threading.ThreadError) as ex:
                raise TaskError(f"Unable to start task of rank {rank}: {ex}")
            self._activity[id_] = None
            self._workers.append((worker, state))

        try:
            feeder = threading.Thread(
                group=None,
                target=self._feed,
                args=(it, iq, oq),
                daemon=True,
            )
            feeder.start()
        except threading.ThreadError as ex:
            raise TaskError(f"Unable to start feeder thread: {ex}")

        # Loop runs on caller thread
        while True:
            try:
                out = self._get(oq)
            except queue.Empty:
                self._check_inactivity()
                continue
            if out is None:
                break
            now = time.time()
            if isinstance(out, logging.LogRecord):
                if self._mtctx == "thread":
                    id_ = out.thread
                else:
                    id_ = out.process
                self._logger.handle(out)
                # Update activity time to indicate that
                # worker is doing something which is generating
                # logs. Do not update if worker has not received
                # a new input yet.
                if self._activity[id_] is not None:
                    self._activity[id_] = now
                self._check_inactivity()
                continue
            id_, typ, value = out
            if typ == LocalParallel._RCVD_MESSAGE:
                # Worker has received the message
                # and starting processing on it
                self._activity[id_] = value
            elif typ == LocalParallel._OUT_MESSAGE:
                # Worker has sent the response back.
                # Set None to indicate that this worker
                # has nothing new to work with to suppress
                # inactivity logs
                self._activity[id_] = None
                yield value
            else:
                raise ValueError(f"Invalid message type received: {typ!r}")
            self._check_inactivity()
        pass

    def _feed(
        self,
        it: Iterable[Any],
        iq: Union[SimpleQueue, Queue],
        oq: Union[SimpleQueue, Queue],
    ):
        it = iter(it)
        self._lastsent = time.time()
        self._inprunning = True
        pipebroke = False
        try:
            while not (pipebroke or self._feedshutdown.is_set()):
                try:
                    item = next(it)
                except StopIteration:
                    # Input exhausted
                    break
                except BrokenPipeError:
                    # If input source is a pipe e.g. stdin then BrokenPipeError
                    # error is expected
                    pipebroke = True
                    break
                except Exception as ex:
                    self._logger.error(
                        f"Error while getting item from input iterable: {ex}"
                    )
                    continue
                if self._put(iq, self._feedshutdown, item):
                    self._lastsent = time.time()
        finally:
            self._inprunning = False

        if pipebroke or self._feedshutdown.is_set():
            return

        # Send kill pill to all workers because
        # input is finished
        for _ in range(len(self._workers)):
            if not self._put(iq, self._feedshutdown, None):
                # Shutdown requested, hence early return
                return

        for worker, _ in self._workers:
            while True:
                if self._feedshutdown.is_set():
                    return
                worker.join(timeout=0.1)
                if worker.is_alive():
                    continue
                break
        LocalParallel._put(oq, self._feedshutdown, None)

    def _ensure_pid(self):
        pid = os.getpid()
        if pid != self._pid:
            raise RuntimeError(
                "Object can only be used from same process which created it"
            )

    def _check_inactivity(self):
        if self._inactivitytimeout == 0.0:
            return
        now = time.time()
        lastsent = self._lastsent
        if self._inprunning and (diff := now - lastsent) > self._inactivitytimeout:
            # Highlight case where producer is slower
            self._logger.warning(f"Nothing sent to workers in last {diff:.3f} seconds")

        for rank, (id_, lastrcvd) in enumerate(self._activity.items()):
            if lastrcvd is None:
                # This worker has not recieved anything new yet
                pass
            elif (diff := now - lastrcvd) > self._inactivitytimeout:
                typ = "tid" if self._mtctx == "thread" else "pid"
                # Highlight case where consumer is slow
                self._logger.warning(
                    f"No activity from rank:{rank}, {typ}:{id_} in last {diff:.3f} seconds"
                )

    def _get(self, q: Union[SimpleQueue, Queue]) -> Any:
        if self._inactivitytimeout == 0.0:
            return q.get(block=True)
        # Wait for 500 millis more to tolerate I/O overhead
        return q.get(timeout=self._inactivitytimeout + 0.5)

    def _stop_threads(self, workers: Iterable[threading.Thread], kbint: bool):
        for worker, state in workers:
            state.setstop()
            if not kbint:
                # Only join if it is not KeyboardInterrupt
                worker.join()
        return

    def _stop_processes(
        self,
        workers: Iterable[multiprocessing.Process],
        manager: Optional[multiprocessing.Manager] = None,
    ):
        for worker, state in workers:
            state.setstop()
            worker.kill()
            worker.join()
        if manager is not None:
            manager.shutdown()

    @staticmethod
    def _run(args: _RunArgs):
        status = 0
        try:
            status = LocalParallel._run_safe(args)
        except KeyboardInterrupt:
            status = -signal.SIGINT
        except BrokenPipeError:
            status = -signal.SIGPIPE
        if args.mtctx != "thread":
            sys.exit(status)

    @staticmethod
    def _run_safe(args: _RunArgs):
        if args.mtctx != "thread":
            # diewithparent(args.ppid)
            pass
        logger = LocalParallel._setup_logger(args.loglevel, args.oq)

        if args.mtctx == "thread":
            # Think of better id_ for thread
            id_ = threading.current_thread().ident
        else:
            id_ = os.getpid()

        try:
            if args.init is not None:
                args.init(args.rank, logger, args.state, *args.initargs)
        except Exception as ex:
            logger.exception(
                f"Exception in rank {args.rank} with id {id_} during init: {ex}"
            )
            if args.mtctx == "thread":
                return
            else:
                sys.exit(1)

        status = 0
        while not args.shutdownevt.is_set():
            try:
                item = args.iq.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if args.sendack:
                if not LocalParallel._put(
                    args.oq,
                    args.shutdownevt,
                    (id_, LocalParallel._RCVD_MESSAGE, time.time()),
                ):
                    break
            try:
                out = args.func(args.state, item)
            except Exception as ex:
                logger.exception(f"Exception in rank: {args.rank} with id {id_}: {ex}")
                continue
            if not LocalParallel._put(
                args.oq, args.shutdownevt, (id_, LocalParallel._OUT_MESSAGE, out)
            ):
                break
        return status

    @staticmethod
    def _setup_logger(loglevel: int, q: Union[SimpleQueue, Queue]) -> logging.Logger:
        if loglevel is None:
            return nolog()

        logger = logging.Logger("LocalParallel")
        # CRITICAL: Prevent logs from bubbling up to parent handlers
        logger.propagate = False
        logger.setLevel(loglevel)
        handler = BlockingQueueHandler(q)
        handler.setLevel(loglevel)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def _put(
        q: Union[SimpleQueue, Queue],
        stop: Union[threading.Event, multiprocessing.Event],
        value: Any,
    ) -> bool:
        added = False
        while not stop.is_set():
            try:
                q.put(value, timeout=0.1)
            except queue.Full:
                continue
            except Exception:
                # To prevent when exception is raised on main thread
                # and things are winding up
                break
            added = True
            break
        return added


__all__ = ["gputasks", "LocalParallel"]
