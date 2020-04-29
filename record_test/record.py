from recordclass import RecordClass, litelist

from record_test.skill import PAType, DrugType, ArrowType, PAParams, PlayerAction


class Star:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class State(RecordClass):
    x: float
    y: float


class Actions(RecordClass):
    idle: PlayerAction = None
    up: PlayerAction = None
    down: PlayerAction = None
    left: PlayerAction = None
    right: PlayerAction = None
    left_up: PlayerAction = None
    left_down: PlayerAction = None
    right_up: PlayerAction = None
    right_down: PlayerAction = None
    shoot: PlayerAction = None
    instant_drug: PlayerAction = None
    continue_drug: PlayerAction = None
    enter_shoot: PlayerAction = None
    exit_shoot: PlayerAction = None
    fire: PlayerAction = None
    escape: PlayerAction = None

    def init(self):
        self.idle = PlayerAction(type=PAType.Idle, name='idle', param=PAParams())
        self.up = PlayerAction(type=PAType.Move, name='up', param=PAParams(direction="up", move_step=(0, 1)))
        self.down = PlayerAction(type=PAType.Move, name='down', param=PAParams(direction="down", move_step=(0, -1)))
        self.left = PlayerAction(type=PAType.Move, name='left', param=PAParams(direction="left", move_step=(-1, 0)))
        self.right = PlayerAction(type=PAType.Move, name='right', param=PAParams(direction="right", move_step=(1, 0)))
        self.left_up = PlayerAction(type=PAType.Move, name='leftup',
                                   param=PAParams(direction="leftup", move_step=(-1, 1)))
        self.left_down = PlayerAction(type=PAType.Move, name='leftdown',
                                     param=PAParams(direction="leftdown", move_step=(-1, 1)))
        self.right_up = PlayerAction(type=PAType.Move, name='rightup',
                                    param=PAParams(direction="rightup", move_step=(1, 1)))
        self.right_down = PlayerAction(type=PAType.Move, name='rightdown',
                                      param=PAParams(direction="rightdown", move_step=(1, -1)))
        self.shoot = PlayerAction(type=PAType.Shoot, name='shoot',
                                  param=PAParams(type=ArrowType.Normal))
        self.instant_drug = PlayerAction(type=PAType.Drug, name='drug instant', param=PAParams(type=DrugType.Instant))
        self.continue_drug = PlayerAction(type=PAType.Drug, name='drug continue',
                                          param=PAParams(type=DrugType.Continue))
        self.enter_shoot = PlayerAction(type=PAType.Status_Switch, name='enter shoot',
                                        param=PAParams(command="enter_shoot"))
        self.exit_shoot = PlayerAction(type=PAType.Status_Switch, name='exit shoot',
                                       param=PAParams(command="exit_shoot"))
        self.fire = PlayerAction(type=PAType.Shoot, name='fire arrow',
                                 param=PAParams(type=ArrowType.Fire))
        self.escape = PlayerAction(type=PAType.Escape, name='escape', param=PAParams())


if __name__ == '__main__':
    action_map = Actions()
    action_map.init()
    for action in list(action_map):
        print(action.name)
    # action_map.idle = PlayerAction(type=PAType.Idle, name='idle', param=PAParams())
    # action_map.left = PlayerAction(type=PAType.Move, name='left', param=PAParams(direction="left", move_step=(-1, 0)))
    print(hasattr(action_map, "xxx"))
    print(action_map.idle.name)
    print(len(action_map))
    print(list(action_map))









