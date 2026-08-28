# agent.py
import random
from collections import deque
import heapq
import math
from logic_engine import KnowledgeBase


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

        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')     # Rule 1
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')  # Rule 2

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

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _tile_facts(self, nx, ny, goal_pos, walls, danger_zones):
        facts = []
        if abs(nx - goal_pos[0]) + abs(ny - goal_pos[1]) <= 6:
            facts.append('TargetVisible')
        adjacent_wall = False
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (nx + dx, ny + dy) in walls:
                adjacent_wall = True
                break
        if adjacent_wall:
            facts.append('HasDust')
        if danger_zones and (nx, ny) in danger_zones:
            facts.append('BloodseekerMissing')
        return facts

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', danger_zones=None):
        sx, sy = start_pos
        gx, gy = goal_pos
        heuristic = (self.manhattan_distance if heuristic_type == 'manhattan'
                     else self.euclidean_distance)

        h_start = heuristic((sx, sy), (gx, gy))
        pq = [(h_start, 0, (sx, sy), [])]        # f = g(0) + h(start)
        reached_states = set()

        dirs = [('Up', 0, 1), ('Right', 1, 0),
                ('Down', 0, -1), ('Left', -1, 0)]
        while pq:
            f_cost, g_cost, (x, y), path = heapq.heappop(pq)
            if (x, y) == (gx, gy):
                return path
            if (x, y) in reached_states:
                continue
            reached_states.add((x, y))
            for name, dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]):
                    continue
                if (nx, ny) in walls:
                    continue
                if (nx, ny) in reached_states:
                    continue

                self.kb.clear_facts()
                for fact in self._tile_facts(nx, ny, (gx, gy), walls, danger_zones):
                    self.kb.tell_fact(fact)
                self.kb.forward_chain()
                if 'Retreat' in self.kb.facts:
                    continue

                g_new = g_cost + 1
                h_new = heuristic((nx, ny), (gx, gy))
                heapq.heappush(pq, (g_new + h_new, g_new, (nx, ny), path + [name]))
        return None

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            grid_size = percept.get('grid_size', (10, 10))
            walls = percept.get('walls', [])
            all_food = percept.get('remaining_food') or percept.get('all_food') or []
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
                elif self.active_algo == 'AStar':
                    self.plan = self.astar_search(start, goal, walls, grid_size) or []
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