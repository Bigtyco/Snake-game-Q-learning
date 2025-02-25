from base import SystemState, AI_Base


class Player(AI_Base):

  def __init__(self):
    self._name = "Human Player"
    self._state = None

  def is_keyboard_allowed(self) -> bool:
    return True

  def callback_take_action(self, state):
    return (state.dir_x, state.dir_y)

  def callback_action_outcome(self, state, outcome):
    pass
