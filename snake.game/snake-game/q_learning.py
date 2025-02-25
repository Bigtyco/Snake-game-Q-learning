import os
import json
import random
import numpy as np
from base import SystemState, AI_Base
from snake import GameOutcome


class AI_RLQ(AI_Base):
  class Action:
    # 动作定义保持不变
    LEFT = 0
    FRONT = 1
    RIGHT = 2
    ALL = [LEFT, FRONT, RIGHT]

    def __init__(self):
      self.action = None

    def __eq__(self, action: int) -> bool:
      return self.action == action

    def __int__(self) -> int:
      return self.action

    def set_action(self, action: int):
      self.action = action

    def get_action(self):
      return self.action

    def to_xy(self, x: int, y: int):
      # 转换逻辑保持不变
      if self.action == self.FRONT:
        pass
      elif self.action == self.LEFT:
        if x != 0:
          y = -x
          x = 0
        else:
          x = y
          y = 0
      elif self.action == self.RIGHT:
        if x != 0:
          y = x
          x = 0
        else:
          x = -y
          y = 0
      return (x, y)

  class State(SystemState):
    def __init__(self, other: SystemState):
      # 原有障碍物检测
      self.obj_front = None
      self.obj_left = None
      self.obj_right = None

      # 新增食物方向检测
      self.food_dir_x = other.food_x - \
          other.head_x if hasattr(other, 'food_x') else 0
      self.food_dir_y = other.food_y - \
          other.head_y if hasattr(other, 'food_y') else 0

      # 原有食物位置检测
      self.food_front = None
      self.food_back = None
      self.food_left = None
      self.food_right = None
      self.dir_x = other.dir_x
      self.dir_y = other.dir_y

      # 方向处理逻辑保持不变
      if other.dir_x == +1:
        self.obj_front = other.obj_east
        self.obj_left = other.obj_north
        self.obj_right = other.obj_south
        self.food_front = other.food_east
        self.food_back = other.food_west
        self.food_left = other.food_north
        self.food_right = other.food_south
      elif other.dir_x == -1:
        self.obj_front = other.obj_west
        self.obj_left = other.obj_south
        self.obj_right = other.obj_north
        self.food_front = other.food_west
        self.food_back = other.food_east
        self.food_left = other.food_south
        self.food_right = other.food_north
      elif other.dir_y == +1:
        self.obj_front = other.obj_south
        self.obj_left = other.obj_east
        self.obj_right = other.obj_west
        self.food_front = other.food_south
        self.food_back = other.food_north
        self.food_left = other.food_east
        self.food_right = other.food_west
      elif other.dir_y == -1:
        self.obj_front = other.obj_north
        self.obj_left = other.obj_west
        self.obj_right = other.obj_east
        self.food_front = other.food_north
        self.food_back = other.food_south
        self.food_left = other.food_west
        self.food_right = other.food_east

    def __eq__(self, other):
      return isinstance(other, SystemState) and str(self) == str(other)

    def __hash__(self):
      return hash(str(self))

    def __str__(self):
      # 增强状态表示
      food_dir = f"({self.food_dir_x},{self.food_dir_y})"
      return f"[{'<' if self.food_left else ' '}" \
             f"{'^' if self.food_front else ' '}" \
             f"{'>' if self.food_right else ' '}" \
             f"{'v' if self.food_back else ' '}]," \
             f"[{self.obj_left},{self.obj_front},{self.obj_right}]," \
             f"{food_dir}"

  def __init__(self, training_mode: bool = True):
    super().__init__()
    self._name = "Q-Learning " + ("" if training_mode else "(testing mode)")

    # 优化训练参数
    self.num_episodes: int = 10000
    self.len_episodes: int = 10000
    self.alpha: float = 0.1
    self.gamma: float = 0.95
    self.epsilon: float = 0.2 if training_mode else 0.0
    self.epsilon_min: float = 0.01
    self.epsilon_decay: float = 0.995
    self.training_mode: bool = training_mode

    self.episode_count = 0

    # 奖励参数
    self.food_reward: int = 10
    self.crash_reward: int = -10
    self.step_penalty: float = -0.1  # 新增移动惩罚

    self.q_table = dict()
    self.current_state = None
    self.current_action = None

    if not self.training_mode:
      self.load_table()

  # 以下辅助方法保持不变
  def load_table(self):
    filename_q_table = "q-table.json"
    if os.path.exists(filename_q_table):
      with open(filename_q_table, "r") as fp:
        self.q_table = json.load(fp)
      print(f"- Loaded {len(self.q_table)} states from {filename_q_table}")

  def save_table(self):
    class NpEncoder(json.JSONEncoder):
      def default(self, obj):
        if isinstance(obj, np.integer):
          return int(obj)
        elif isinstance(obj, np.floating):
          return float(obj)
        elif isinstance(obj, np.ndarray):
          return obj.tolist()
        return super().default(obj)

    with open("q-table.json", "w") as fp:
      json.dump(self.q_table, fp, cls=NpEncoder, indent=4)

  def q(self, state):
    s = str(state)
    if s not in self.q_table:
      self.q_table[s] = np.zeros(len(self.Action.ALL))
    return self.q_table[s]

  def callback_take_action(self, state: SystemState):
    s = self.State(state)
    self.current_state = s

    possible_actions = []
    # 安全动作过滤
    for action in self.Action.ALL:
      obj_status = {
          self.Action.FRONT: s.obj_front,
          self.Action.LEFT: s.obj_left,
          self.Action.RIGHT: s.obj_right
      }[action]
      if obj_status > 0:  # 只允许安全方向
        possible_actions.append(action)

    # 保底处理：如果没有安全动作，允许所有动作
    if not possible_actions:
      possible_actions = self.Action.ALL.copy()

    # Epsilon-greedy 策略
    if random.uniform(0, 1) < self.epsilon:
      chosen_action = random.choice(possible_actions)
    else:
      # 从安全动作中选择最优
      q_values = self.q(s)
      valid_q = [q_values[a] for a in possible_actions]
      max_q = max(valid_q)
      best_actions = [a for a in possible_actions if q_values[a] == max_q]
      chosen_action = random.choice(best_actions)

    a = self.Action()
    a.set_action(chosen_action)
    self.current_action = a

    return a.to_xy(s.dir_x, s.dir_y)

  def callback_action_outcome(self, state: SystemState, outcome: GameOutcome):
    s = self.current_state
    a = self.current_action
    s1 = self.State(state)

    # 奖励计算
    if outcome in [GameOutcome.CRASHED_TO_BODY, GameOutcome.CRASHED_TO_WALL]:
      reward = self.crash_reward
    elif outcome == GameOutcome.REACHED_FOOD:
      reward = self.food_reward
    else:
      reward = self.step_penalty  # 新增移动惩罚

    if self.training_mode:
      # Q-learning 更新规则
      a_idx = int(a)
      current_q = self.q(s)[a_idx]
      max_future_q = np.max(self.q(s1))
      new_q = current_q + self.alpha * \
          (reward + self.gamma * max_future_q - current_q)
      self.q(s)[a_idx] = new_q

  def callback_episode_end(self, episode, steps, outcome):
    """新增的episode结束回调"""
    if self.training_mode:
        # 执行epsilon衰减
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
      self.episode_count += 1

      # 每100个episode保存一次
      if self.episode_count % 100 == 0:
        self.save_table()

  def callback_terminating(self):
    if self.training_mode:
      # Epsilon衰减
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
      self.save_table()
