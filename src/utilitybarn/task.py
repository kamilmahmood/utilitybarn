import logging
import multiprocessing
import multiprocessing.process
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from typing import Any, Callable, Iterable, List, Literal, NoReturn, Optional, Union

from packaging import version

from .errors import NoAcceleratorFoundError, TaskError
from .log import BlockingQueueHandler, nolog


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


class _ParallelExecutorBase(object):
    """Base class for parallel executors e.g.
    ProcessParallel and ThreadParallel"""

    def __init__(self, backend, inactivitytimeout: float, logger: logging.Logger):
        super().__init__()
        if type(self) is _ParallelExecutorBase:
            raise TypeError(
                f"{_ParallelExecutorBase.__class__.__name__} is an abstract base class and cannot be"
                " instantiated directly. Use ProcessParallel or ThreadParallel instead."
            )
        self._backend = backend
        self._inactivitytimeout = inactivitytimeout
        self._logger = logger

        # Mutable state
        self._workers = None
        self._activity = None
        self._lastsent = None
        self._inprunning = False

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
                typ = "tid" if self._backend == "mt" else "pid"
                # Highlight case where consumer is slow
                self._logger.warning(
                    f"No activity from rank:{rank}, {typ}:{id_} in last {diff:.3f} seconds"
                )

    def _feed(
        self, it: Iterable[Any], iq: Union[Queue, queue.Queue], oq: [Queue, queue.Queue]
    ):
        it = iter(it)
        self._inprunning = True
        self._lastsent = time.time()
        while True:
            try:
                item = next(it)
            except StopIteration:
                break
            except Exception as ex:
                self._logger.error(f"Error while getting item from iterable: {ex}")
                continue

            if self._inactivitytimeout != 0.0:
                while True:
                    try:
                        # Timeout 500 millis before actual time so that last sent is
                        # set to slighlty before threshold to prevent extra warnings
                        iq.put(item, timeout=max(0.1, self._inactivitytimeout - 0.5))
                    except queue.Full:
                        # Queue is full at the moment, try next time
                        pass
                    else:
                        # Item successfully put into queue
                        break
                    finally:
                        self._lastsent = time.time()
            else:
                iq.put(item, block=True)
        self._inprunning = False

        # Send kill pill to all workers because
        # input is finished
        for w in range(len(self._workers)):
            iq.put(None)

        for worker in self._workers:
            worker.join()
        oq.put(None, block=True)

    def _cleanup(
        self, manager: Optional[multiprocessing.Manager] = None, kbint: bool = False
    ):
        if self._workers is not None:
            for worker in self._workers:
                if self._backend == "mp":
                    worker.kill()
                    worker.join()
                elif self._backend == "mt":
                    # TODO: Fix leaky abstraction where parent is concerned
                    # about child behaviour
                    worker._utilitybarn_thread_worker_state._shoudstop = True
                    if not kbint:
                        # Only join if it is not KeyboardInterrupt
                        worker.join()
        if manager is not None:
            manager.shutdown()
        self._workers = None
        self._activity = None
        self._lastsent = None
        self._inprunning = False


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


class ProcessParallel(_ParallelExecutorBase):
    """
    Run tasks concurrently using multiple processes with controlled buffering,
    safe logging, and activity monitoring.

    This class provides functionality similar to `multiprocessing.Pool` and
    `concurrent.futures.ProcessPoolExecutor`, but with several key differences:

    1. **Bounded Buffers**
       Input (`inpbuffsize`) and output (`outbuffsize`) queues are size limited
       to reduce memory pressure and handle cases where producer and consumer
       speeds differ significantly.

    2. **Multiprocessing-Safe Logging**
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

    Overall, `ProcessParallel` is designed for workloads where memory
    efficiency, reliable logging, hardware-aware scheduling, and responsiveness
    to stalled workers are critical.

    Parameters
    ----------
    njobs : int
        Number of worker processes to launch. Each worker runs tasks independently.
        Example: `njobs=4` will use 4 processes concurrently.
    init : Callable[[int, logging.Logger, ...], Any], optional
        Optional initialization function called once per worker before processing
        begins. Receives the worker's rank as first argument and
        a multiprocessing-safe logger. Useful for setting up resources such as
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
    mpctx : multiprocessing context, optional
        Multiprocessing context (`fork`, `spawn`, or `forkserver`). If not provided,
        the system default is used. Choose explicitly if you need predictable
        behavior across platforms.
    inactivitytimeout : float, default=0.0
        Maximum allowed inactivity (in seconds) for workers or producer before a
        warning is logged. Set to >0.0 to detect stalls in input or output flow.

    Example
    -------
    >>> from utilitybarn.log import stderr
    >>> from utilitybarn.task import ProcessParallel, gputasks
    >>>
    >>> # Global variables for process state
    >>> LOGGER = None
    >>> RANK = None
    >>>
    >>> def init(rank, logger, *args):
    >>>     global RANK, LOGGER
    >>>     RANK = rank
    >>>     LOGGER = logger
    >>>     gpus = args[0]
    >>>     logger.info(f"Worker with rank {rank} Using GPU {gpus[rank]}")
    >>>
    >>>
    >>> def mul(args):
    >>>     a, b = args
    >>>     LOGGER.info(f"Multiplying {a} and {b}")
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
        init: Optional[Callable[[int, logging.Logger, ...], Any]] = None,
        initargs: Optional[tuple[Any]] = None,
        inpbuffsize: int = 64,
        outbuffsize: int = 64,
        logger: logging.Logger = nolog(),
        mpctx: Union[BaseContext, str, None] = None,
        inactivitytimeout: float = 0.0,
    ):
        super().__init__("mp", inactivitytimeout, logger)
        if njobs <= 0:
            raise ValueError(f":njobs cannot be <1 but got {njobs}")
        self._njobs = njobs
        self._init = init
        self._initargs = initargs or ()
        self._inpbuffsize = inpbuffsize
        self._outbuffsize = outbuffsize
        if mpctx is None:
            startmethod = multiprocessing.get_start_method()
            self._mpctx = multiprocessing.get_context(startmethod)
        elif isinstance(mpctx, str):
            self._mpctx = multiprocessing.get_context(mpctx)
        else:
            self._mpctx = mpctx
        self._pid = os.getpid()
        self._lock = threading.Lock()
        # Mutable state
        self._running = False
        self._iq = None
        self._oq = None
        self._manager = None

    def apply(self, func: Callable[[Any], Any], it: Iterable[Any]) -> Iterable[Any]:
        """Run a `func` in parallel across multiple processes and consume results
        as they become available.

        This method is designed to distribute work items from an iterable to
        several worker processes and collect their outputs.
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
        >>> from utilitbarn.task import ProcessParallel
        >>> pp = ProcessParallel(njobs=4)
        >>> results = pp.apply(lambda x: x * x, range(10))
        >>> for r in results:
        >>>     print(r)
        # Output will contain squares of 0-9, but not necessarily in order.
        """

        self._ensure_pid()
        with self._lock:
            if self._running:
                raise RuntimeError("Already running")
            self._running = True

        try:
            # Only create manager when processing is running
            # TODO: Better cleanup in case of crash or kill signal
            self._manager = self._mpctx.Manager()
            yield from self._apply(func, it)
        finally:
            self._cleanup(manager=self._manager)
            self._iq = None
            self._oq = None
            with self._lock:
                self._running = False
                self._manager = None

    def _apply(self, func: Callable[Any, Any], it: Iterable[Any]) -> Iterable[Any]:
        self._iq = self._manager.Queue(self._inpbuffsize)
        self._oq = self._manager.Queue(self._outbuffsize)
        self._workers = []
        loglevel = self._logger.level if self._logger.hasHandlers() else None
        for rank in range(self._njobs):
            try:
                worker = self._mpctx.Process(
                    target=ProcessParallel._run,
                    name=None,
                    args=(
                        self._pid,
                        rank,
                        loglevel,
                        self._inactivitytimeout != 0.0,
                        self._iq,
                        self._oq,
                        self._init,
                        self._initargs,
                        func,
                    ),
                    daemon=True,
                )
                worker.start()
            except multiprocessing.ProcessError as ex:
                raise TaskError(f"Unable to start process of rank {rank}: {ex}")
            self._workers.append(worker)

        try:
            feeder = threading.Thread(
                group=None,
                target=self._feed,
                args=(it, self._iq, self._oq),
                daemon=True,
            )
            feeder.start()
        except threading.ThreadError as ex:
            raise TaskError(f"Unable to start feeder thread: {ex}")

        self._activity = OrderedDict([(worker.pid, None) for worker in self._workers])
        while True:
            if self._inactivitytimeout != 0.0:
                try:
                    # Wait for 500 millis more to tolerate I/O overhead
                    out = self._oq.get(timeout=self._inactivitytimeout + 0.5)
                except queue.Empty:
                    self._check_inactivity()
                    continue
            else:
                out = self._oq.get(block=True)
            if out is None:
                break
            now = time.time()
            if isinstance(out, logging.LogRecord):
                pid = out.process
                self._logger.handle(out)
                # Update activity time to indicate that
                # worker is doing something which is generating
                # logs. Do not update if worker has not received
                # a new input yet.
                if self._activity[pid] is not None:
                    self._activity[pid] = now
            else:
                pid, typ, value = out
                if typ == ProcessParallel._RCVD_MESSAGE:
                    # Worker has received the message
                    # and starting processing on it
                    self._activity[pid] = value
                elif typ == ProcessParallel._OUT_MESSAGE:
                    # Worker has sent the response back.
                    # Set None to indicate that this worker
                    # has nothing new to work with to suppress
                    # inactivity logs
                    self._activity[pid] = None
                    yield value
                else:
                    raise ValueError(f"Invalid message type received: {typ!r}")
            self._check_inactivity()

    def _ensure_pid(self):
        pid = os.getpid()
        if pid != self._pid:
            raise RuntimeError(
                "Object can only be used from same process which created it"
            )

    @staticmethod
    def _run(
        ppid: int,
        rank: int,
        loglevel: Optional[int],
        sendack: bool,
        iq: Queue,
        oq: Queue,
        init: Optional[Callable[[tuple[Any]], Any]],
        initargs: tuple[Any],
        func: Callable[[tuple[Any]], Any],
    ) -> NoReturn:
        ProcessParallel._setup_parent_death_action(ppid)
        if os.getppid() != ppid:
            """Kill it if process was reparented before installing
            death action for parent"""
            sys.exit(0)
        # TODO: Is there a need to setup some signal handlers?

        logger = ProcessParallel._setup_logger(loglevel, oq)
        try:
            if init is not None:
                init(rank, logger, *initargs)
        except Exception as ex:
            logger.exception(
                f"Exception in rank {rank} with pid {os.getpid()} during init: {ex}"
            )
            sys.exit(1)

        pid = os.getpid()
        while True:
            item = iq.get(block=True)
            if item is None:
                break
            if sendack:
                oq.put((pid, ProcessParallel._RCVD_MESSAGE, time.time()), block=True)
            try:
                out = func(item)
            except Exception as ex:
                logger.exception(f"Exception in rank: {rank} with pid {pid}: {ex}")
                continue
            oq.put((pid, ProcessParallel._OUT_MESSAGE, out), block=True)
        sys.exit(0)

    @staticmethod
    def _setup_parent_death_action(ppid):
        # TODO
        pass

    @staticmethod
    def _setup_logger(loglevel, q) -> logging.Logger:
        if loglevel is None:
            return nolog()
        logger = logging.Logger(__name__)
        logger.setLevel(loglevel)
        # Remove all exsiting handlers so message
        # is not routed anywhere unexpected
        for h in logger.handlers.copy():
            logger.removeHandler(h)
        handler = BlockingQueueHandler(q)
        handler.setLevel(loglevel)
        logger.addHandler(handler)
        return logger


class ThreadState(object):
    """
    Object of this class will be passed
    worker threads to keep their state
    """

    def __init__(self):
        super().__init__()
        self._shoudstop = False

    def shouldstop(self) -> bool:
        """Check whether worker thread should stop or not.
        It is to support collaborative terminate functionality
        for thread
        """
        return self._shoudstop


class ThreadParallel(_ParallelExecutorBase):
    """
    Run tasks concurrently using multiple threads with controlled buffering,
    and activity monitoring.

    This class provides functionality similar to `multiprocessing.pool.ThreadPool` and
    `concurrent.futures.ThreadPoolExecutor`, but with several key differences:

    1. **Bounded Buffers**
       Input (`inpbuffsize`) and output (`outbuffsize`) queues are size limited
       to reduce memory pressure and handle cases where producer and consumer
       speeds differ significantly.

    2. **Rank Awareness**
       Each worker is assigned a number (rank), enabling scenarios
       where workers must bind to specific hardware resources (e.g., GPUs).

    3. **Inactivity Monitoring**
       Both workers and the producer are monitored for inactivity via
       `inactivitytimeout`. Warnings are issued when input or output stalls,
       helping detect bottlenecks or deadlocks early.

    Parameters
    ----------
    njobs : int
        Number of worker threads to launch. Each worker runs tasks independently.
        Example: `njobs=4` will use 4 threads concurrently.
    init : Callable[[int, logging.Logger, ThreadState, ...], Any], optional
        Optional initialization function called once per worker before processing
        begins. Receives the worker's rank as first argument, logger and state object.
        New attributes can added to state and it will be passed again with each input.
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
        Logger instance used by workers. Default logger logs are not printed.
    inactivitytimeout : float, default=0.0
        Maximum allowed inactivity (in seconds) for workers or producer before a
        warning is logged. Set to >0.0 to detect stalls in input or output flow.

    Example
    -------
    >>> from utilitybarn.log import stderr
    >>> from utilitybarn.task import ThreadParallel
    >>>
    >>>
    >>> def init(rank, logger, state):
    >>>     state.rank = rank
    >>>     state.logger = logger
    >>>     logger.info(f"Worker with rank {rank}")
    >>>
    >>>
    >>> def mul(state, args):
    >>>     a, b = args
    >>>     state.logger.info(f"Multiplying {a} and {b} in rank:{state.rank}")
    >>>     return a * b
    >>>
    >>>
    >>> p = ThreadParallel(njobs=4, init=init, logger=stderr())
    >>> for res in p.apply(mul, [(20, 30), (40, 50)]):
    >>>     print(res)
    >>>

    Limitations
    -----------
    1. Workers which crashes are not restarted
       and their inactivity time is not updated.
    2. Uses collaborative way to stop worker thread
       instead of preemptive.
    """

    _RCVD_MESSAGE = "R"
    _OUT_MESSAGE = "O"

    def __init__(
        self,
        *,
        njobs: int,
        init: Optional[Callable[[int, logging.Logger, ThreadState, ...], Any]] = None,
        initargs: Optional[tuple[Any]] = None,
        inpbuffsize: int = 64,
        outbuffsize: int = 64,
        logger: logging.Logger = nolog(),
        inactivitytimeout: float = 0.0,
    ):
        super().__init__("mt", inactivitytimeout, logger)
        if njobs <= 0:
            raise ValueError(f":njobs cannot be <1 but got {njobs}")
        self._njobs = njobs
        self._init = init
        self._initargs = initargs or ()
        self._inpbuffsize = inpbuffsize
        self._outbuffsize = outbuffsize
        self._lock = threading.Lock()

        # Mutable state
        self._running = False
        self._iq = None
        self._oq = None

    def apply(
        self, func: Callable[[ThreadState, Any], Any], it: Iterable[Any]
    ) -> Iterable[Any]:
        """Run a `func` in parallel across multiple threads and consume results."""
        with self._lock:
            if self._running:
                raise RuntimeError("Already running")
            self._running = True

        kbint = False
        try:
            yield from self._apply(func, it)
        except KeyboardInterrupt:
            kbint = True
            raise
        finally:
            self._cleanup(kbint=kbint)
            self._iq = None
            self._oq = None
            with self._lock:
                self._running = False

    def _apply(
        self, func: Callable[[ThreadState, Any], Any], it: Iterable[Any]
    ) -> Iterable[Any]:
        self._iq = queue.Queue(self._inpbuffsize)
        self._oq = queue.Queue(self._outbuffsize)
        self._workers = []

        for rank in range(self._njobs):
            state = ThreadState()
            try:
                worker = threading.Thread(
                    target=ThreadParallel._run,
                    args=(
                        rank,
                        self._logger,
                        self._inactivitytimeout != 0.0,
                        self._iq,
                        self._oq,
                        self._init,
                        self._initargs,
                        func,
                    ),
                    daemon=True,
                )
                worker._utilitybarn_thread_worker_state = state
                worker.start()
            except threading.ThreadError as ex:
                raise TaskError(f"Unable to start thread of rank {rank}: {ex}")
            self._workers.append(worker)

        try:
            feeder = threading.Thread(
                target=self._feed,
                args=(it, self._iq, self._oq),
                daemon=True,
            )
            feeder.start()
        except threading.ThreadError as ex:
            raise TaskError(f"Unable to start feeder thread: {ex}")

        self._activity = OrderedDict([(w.native_id, None) for w in self._workers])
        while True:
            try:
                if self._inactivitytimeout != 0.0:
                    out = self._oq.get(timeout=self._inactivitytimeout + 0.5)
                else:
                    out = self._oq.get(block=True)
            except queue.Empty:
                self._check_inactivity()
                continue

            if out is None:
                break
            pid, typ, value = out
            if typ == ThreadParallel._RCVD_MESSAGE:
                self._activity[pid] = value
            elif typ == ThreadParallel._OUT_MESSAGE:
                self._activity[pid] = None
                yield value
            else:
                raise ValueError(f"Invalid message type received: {typ!r}")
            self._check_inactivity()

    @staticmethod
    def _run(
        rank: int,
        logger: logging.Logger,
        sendack: bool,
        iq: queue.Queue,
        oq: queue.Queue,
        init: Optional[Callable[[int, logging.Logger, ThreadState, ...], Any]],
        initargs: tuple[Any],
        func: Callable[[ThreadState, Any], Any],
    ):
        state = threading.current_thread()._utilitybarn_thread_worker_state
        try:
            if init is not None:
                init(rank, logger, state, *initargs)
        except Exception as ex:
            logger.exception(f"Exception in rank {rank} during init: {ex}")
            return

        tid = threading.current_thread().native_id
        while True:
            try:
                item = iq.get(timeout=0.5)
            except queue.Empty:
                if state.shouldstop():
                    break
                continue
            if item is None:
                break
            if sendack:
                oq.put((tid, ThreadParallel._RCVD_MESSAGE, time.time()), block=True)
            try:
                out = func(state, item)
            except Exception as ex:
                logger.exception(f"Exception in rank {rank}, thread {tid}: {ex}")
                continue
            oq.put((tid, ThreadParallel._OUT_MESSAGE, out), block=True)
        return


__all__ = ["gputasks", "ProcessParallel"]
