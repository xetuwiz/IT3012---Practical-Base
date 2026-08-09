# agent.py
import random


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Up'
        if percept.get('wall_ahead'):
            return 'Left'
        return 'Up'


class ModelBasedAgent:
    def __init__(self):
        self.moves = ['Up', 'Right', 'Down', 'Left']
        self.dx = [0, 1, 0, -1]
        self.dy = [1, 0, -1, 0]
        self.x = 0
        self.y = 0
        self.facing = 0
        self.last_action = None
        self.stuck = 0
        self.visited_cells = set()
        self.visited_cells.add((self.x, self.y))

    def _update_model(self):
        if self.last_action is not None:
            if self.last_action in self.moves:
                self.facing = self.moves.index(self.last_action)
        self.visited_cells.add((self.x, self.y))

    def sense_and_act(self, percept: dict) -> str:
        self._update_model()

        if percept.get('food_here'):
            act = self.moves[self.facing]
        elif percept.get('wall_ahead'):
            self.stuck += 1
            if self.stuck % 2 == 1:
                act = self.moves[(self.facing - 1) % 4]
            else:
                act = self.moves[(self.facing + 1) % 4]
        else:
            self.x += self.dx[self.facing]
            self.y += self.dy[self.facing]
            act = self.moves[self.facing]

        self.last_action = act
        self.visited_cells.add((self.x, self.y))
        return act