from utilitybarn.task import LocalParallel


class TestLocalParallel:
    def test_thread_parallel(self):
        tp = LocalParallel(
            njobs=15,
            mtctx="thread",
            init=TestLocalParallel._init,
            initargs=(None, None),
        )
        inp = [1, 2, 3]
        res = list(tp.apply(TestLocalParallel._run, inp))
        assert res == inp

    def test_fork_parallel(self):
        tp = LocalParallel(
            njobs=15,
            mtctx="fork",
            init=TestLocalParallel._init,
            initargs=(None, None),
        )
        inp = [1, 2, 3]
        res = list(tp.apply(TestLocalParallel._run, inp))
        assert sorted(res) == inp

    def test_spawn_parallel(self):
        tp = LocalParallel(
            njobs=15,
            mtctx="spawn",
            init=TestLocalParallel._init,
            initargs=(None, None),
        )
        inp = [1, 2, 3]
        res = list(tp.apply(TestLocalParallel._run, inp))
        assert sorted(res) == inp

    def test_forkserver_parallel(self):
        tp = LocalParallel(
            njobs=15,
            mtctx="forkserver",
            init=TestLocalParallel._init,
            initargs=(None, None),
        )
        inp = [1, 2, 3]
        res = list(tp.apply(TestLocalParallel._run, inp))
        assert sorted(res) == inp

    @staticmethod
    def _init(rank, state, logger, *args):
        pass

    @staticmethod
    def _run(rank, args):
        return args


__all__ = ["TestLocalParallel"]
