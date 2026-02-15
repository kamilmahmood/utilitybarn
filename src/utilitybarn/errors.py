class UtilityBarnError(Exception):
    """Base class for all errors raised
    from utility barn
    """

    pass


class TaskError(UtilityBarnError):
    """Error raised when there is something
    wrong with Process Parallel or Thread Parallel
    """

    pass


class NoAcceleratorFoundError(TaskError):
    """Error raised when there is no required
    accelerator found
    """

    pass


__all__ = ["UtilityBarnError", "TaskError", "NoAcceleratorFoundError"]
