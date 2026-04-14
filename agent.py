import config
import numpy as np

from brain import Brain, BrainConfig
from world import TileType
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4
    TALK = 5

def encode_tile(tile_type: TileType):
    if tile_type == TileType.VOID:
        return [1.0, 0.0, 0.0, 0.0]
    elif tile_type == TileType.FOOD:
        return [0.0, 1.0, 0.0, 0.0]
    elif tile_type == TileType.AGENT:
        return [0.0, 0.0, 1.0, 0.0]
    else: 
        return [0.0, 0.0, 0.0, 1.0]

class Agent:
    def __init__(self, id: int, spawn_loc: tuple[int, int], brain_cfg: BrainConfig):
        self.id = id
        self.loc = spawn_loc
        self.brain = Brain(brain_cfg)
        self.satiety = config.AGENT_SATIETY

    def perform(self, map: np.typing.NDarray[object]):
        state = self.get_state(map)
        actions = self.brain.forward(state)
        action: Action = np.argmax(actions)

        reward = -0.1
        done = False
        
        if action == Action.EAT:
            food_loc = self.check_adjacent_food(map)

            if food_loc:
                fy, fx = food_loc
                map[fy, fx].type = TileType.VOID
                self.eat()
                reward = 10.0
            else:
                reward = -2.0 
        else:
            success = self.move_agent(action, map)

            if not success:
                reward = -2.0
        
        self.satiety -= 0.2

        if self.satiety <= 0:
            reward = -10.0
            done = True

        return done

    def check_adjacent_food(self, map: np.typing.NDarray[object]):
        y, x = self.loc

        surrounding_coords = [
            (y-1, x), (y+1, x), (y, x-1), (y, x+1),
            (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
        ]

        map_size = map.shape
        
        for y, x in surrounding_coords:
            if 0 <= y < map_size[0] and 0 <= x < map_size[1]:
                if map[y, x].type == TileType.FOOD:
                    return (y, x)
        return None

    def get_state(self, map: np.typing.NDarray[object]):
        state = []
        
        y, x = self.loc
        
        surrounding_coords = [
            (y-1, x), (y+1, x), (y, x-1), (y, x+1),
            (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
        ]

        map_size = map.shape
        
        for y, x in surrounding_coords:
            if 0 <= y < map_size[0] and 0 <= x < map_size[1]:
                tile = map[y, x]
                state.extend(encode_tile(tile.type))
            else:
                state.extend([0.0, 0.0, 0.0, 1.0])

        state.append(self.satiety)
        return state

    def eat(self):
        self.satiety += 1

    def move(self, action: Action, map: np.typing.NDarray[object]):
        y, x = self.loc
        ny, nx = y, x
        
        if action == Action.UP: ny -= 1
        elif action == Action.DOWN: ny += 1
        elif action == Action.LEFT: nx -= 1
        elif action == Action.RIGHT: nx += 1
        
        map_size = map.shape
        
        if 0 <= ny < map_size[0] and 0 <= nx < map_size[1]:
            target_tile = map[ny, nx]
            if target_tile.type == TileType.VOID:
                map[y, x].type = TileType.VOID
                map[y, x].agent = None
                
                target_tile.type = TileType.AGENT
                target_tile.agent = self 
                self.loc = (ny, nx)
                return True

        return False

    
