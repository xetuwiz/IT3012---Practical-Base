# agent.py
import random
from collections import deque
import heapq


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
    
class SearchAgent:
    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'
        self.x = 0
        self.y = 0

    def bfs_search(self, start, goal, walls, grid_size):
        sx, sy = start
        gx, gy = goal
        q = deque()
        q.append((sx, sy, []))          # (x, y, path_so_far)
        reached = set()
        reached.add((sx, sy))
        
        dirs = [('Up', 0, 1), ('Right', 1, 0),
                ('Down', 0, -1), ('Left', -1, 0)]
        while q:
            x, y, path = q.popleft()
            if (x, y) == (gx, gy):
                return path
            for name, dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]
                        and (nx, ny) not in walls and (nx, ny) not in reached):
                    reached.add((nx, ny))
                    q.append((nx, ny, path + [name]))
        return None
    
    def dfs_search(self, start, goal, walls, grid_size):
        sx, sy = start
        gx, gy = goal
        stack = [(sx, sy, [])]          # (x, y, path_so_far)
        reached = set()
        reached.add((sx, sy))
        dirs = [('Up', 0, 1), ('Right', 1, 0),
                ('Down', 0, -1), ('Left', -1, 0)]
        while stack:
            x, y, path = stack.pop()
            if (x, y) == (gx, gy):
                return path
            for name, dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]
                        and (nx, ny) not in walls and (nx, ny) not in reached):
                    reached.add((nx, ny))
                    stack.append((nx, ny, path + [name]))
        return None
    
    def ucs_search(self, start, goal, walls, grid_size):
        sx, sy = start
        gx, gy = goal
        # priority queue entries: (cost_so_far, x, y, path_so_far)
        pq = [(0, sx, sy, [])]
        reached = set()
        reached.add((sx, sy))
        dirs = [('Up', 0, 1), ('Right', 1, 0),
                ('Down', 0, -1), ('Left', -1, 0)]
        while pq:
            cost, x, y, path = heapq.heappop(pq)
            if (x, y) == (gx, gy):
                return path
            for name, dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]
                        and (nx, ny) not in walls and (nx, ny) not in reached):
                    reached.add((nx, ny))
                    heapq.heappush(pq, (cost + 1, nx, ny, path + [name]))
        return None

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            grid_size = percept.get('grid_size', (10, 10))
            walls = percept.get('walls', [])
            all_food = percept.get('all_food', [])
            if all_food:
                closest = min(
                    all_food,
                    key=lambda f: abs(f[0] - self.x) + abs(f[1] - self.y))
                goal = (closest[0], closest[1])
                start = (self.x, self.y)
                if self.active_algo == 'BFS':
                    self.plan = self.bfs_search(start, goal, walls, grid_size) or []
                elif self.active_algo == 'DFS':
                    self.plan = self.dfs_search(start, goal, walls, grid_size) or []
                else:  # UCS
                    self.plan = self.ucs_search(start, goal, walls, grid_size) or []

        if self.plan:
            action = self.plan.pop(0)
            self._update_position(action)
            return action
        return 'Up'

    def _update_position(self, action: str):
        if action == 'Up':
            self.y += 1
        elif action == 'Down':
            self.y -= 1
        elif action == 'Right':
            self.x += 1
        elif action == 'Left':
            self.x -= 1