import time
import os
import config
import numpy as np

from world import World
from enumdefs import TileType

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_map(world_map: np.typing.NDarray[object]):
    symbol_map = {
        TileType.VOID: '.',
        TileType.FOOD: '*',
        TileType.AGENT: '@'
    }
    
    rows, cols = world_map.shape
    for r in range(rows):
        row_str = " ".join(symbol_map[world_map[r, c].type] for c in range(cols))
        print(row_str)

def run_simulation(steps: int = 200, delay: float = 0.3):
    world = World(
        map_size=config.MAP_SIZE,
        seed=config.SEED,
        num_agents=config.NUM_AGENTS,
        num_food=config.NUM_FOOD
    )

    for step in range(steps):
        clear_screen()
        print(f"--- Step {step + 1}/{steps} ---")

        if step > 0 and step % config.FOOD_SPAWN_INTERVAL == 0:
            world.replenish_food(config.FOOD_SPAWN_AMOUNT)
        
        agents = []
        for row in world.map:
            for tile in row:
                if tile.type == TileType.AGENT and tile.agent is not None:
                    agents.append(tile.agent)

        total_loss = 0.0
        for agent in agents:
            loss = agent.perform(world.map)
            if loss is not None:
                total_loss += loss

        print_map(world.map)
        
        active_agents = sum(1 for row in world.map for t in row if t.type == TileType.AGENT)
        remaining_food = sum(1 for row in world.map for t in row if t.type == TileType.FOOD)
        avg_loss = total_loss / len(agents) if agents else 0.0
        
        print(f"\nActive Agents: {active_agents}")
        print(f"Remaining Food: {remaining_food}")
        print(f"Average Loss:  {avg_loss:.4f}")
        
        if active_agents == 0:
            print("\nAll agents have died. Simulation terminated.")
            break
            
        time.sleep(delay)

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")