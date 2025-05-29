class Cache:
    @staticmethod
    def enable_cache(_):
        pass

def get_session(year, round, session):
    class Dummy:
        def load(self):
            pass
    return Dummy()
