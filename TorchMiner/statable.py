class Statable:
    def load_state_dict(self, state):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError
