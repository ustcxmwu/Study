from enum import Enum


class PAType(Enum):
    Idle = "idle"
    Move = "move"
    Shoot = "shoot"
    Drug = "drug"
    Status_Switch = "status_switch"
    Escape = "escape"


class DrugType(Enum):
    Instant = 0
    Continue = 1


class ArrowType(Enum):
    Normal = 0
    Fire = 1
    Ice = 2


class PAParams(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, item: str, default=None):
        return getattr(self, item, default)


class PAStatus(object):
    def __init__(self):
        pass

    def update(self):
        pass


class PlayerAction(object):
    """ Player Action

    Attributes:
        type: action type
        name: action name
        cd: action cd time in seconds
        read: read the status, designed for drug
        param:

    """

    def __init__(self, type: PAType, name: str, cd: float = 0.0, read: float = 0.0, param: PAParams = None):
        self._type = type
        self._name = name
        self._cd = cd
        self._read = read
        self._param = param

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def cd(self):
        return self._cd

    @property
    def read(self):
        return self._read

    @property
    def param(self):
        return self._param

    def update_param(self, **kwargs):
        self._param.update(**kwargs)
