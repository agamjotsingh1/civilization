import numpy as np
import itertools
from enum import Enum, auto

class TileType(Enum):
    VOID = auto()
    FOOD = auto()
    AGENT = auto()

class Tile:
    id_generator = itertools.count(1)

    def __init__(self, tile_type: TileType):
        self.type = tile_type
        
        if self.type == TileType.VOID:
            self.id = 0
        else:
            self.id = next(self.id_generator)

    def __repr__(self):
        return f"{self.type.name}_{self.id}"

class World:
    def __init__(self, map_size: tuple[int, int], seed: int, num_agents: int, num_food: int):
        self.map_size = map_size
        self.seed = seed
        self.num_agents = num_agents
        self.num_food = num_food
        self.rng = np.random.default_rng(self.seed)
        
        self.init_map()

    def init_map(self):
        rows, cols = self.map_size
        
        self.map = np.array(
            [[Tile(TileType.VOID) for _ in range(cols)] for _ in range(rows)], 
            dtype=object
        )
        
        self.spawn_tiles(self.num_agents, TileType.AGENT)
        self.spawn_tiles(self.num_food, TileType.FOOD)

    def spawn_tiles(self, count: int, tile_type: TileType):
        is_void = np.vectorize(lambda tile: tile.type == TileType.VOID)
        empty_indices = np.argwhere(is_void(self.map))
        
        if len(empty_indices) < count:
            raise ValueError(f"Cannot spawn {count} items; only {len(empty_indices)} spaces available.")

        chosen_indices = self.rng.choice(len(empty_indices), size=count, replace=False)
        coords = empty_indices[chosen_indices]

        for y, x in coords:
            self.map[y, x] = Tile(tile_type)

if __name__ == "__main__":
    world = World(map_size=(9, 9), seed=42, num_agents=10, num_food=20)
    print(world.map)