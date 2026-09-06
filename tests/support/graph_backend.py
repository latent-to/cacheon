"""CPU rehearsal of verifier sequencing; never CUDA hardware evidence."""

class FakeGraphBackend:
    """CPU model of capture/replay orchestration (not CUDA semantics themselves)."""

    def __init__(self):
        self.phase = "eager"
        self.replay_index = -1

    def warmup(self, fn):
        self.phase = "warmup"
        fn()
        self.phase = "eager"

    def capture(self, fn):
        self.phase = "capture"
        fn()
        self.phase = "eager"
        return fn

    def replay(self, graph):
        self.replay_index += 1
        self.phase = "replay"
        graph()
        self.phase = "eager"

    def synchronize(self):
        pass
