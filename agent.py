from brain import Brain

class Agent:
    def __init__(self, id, spawn_loc, brain_cfg):
        self.id = id
        self.loc = spawn_loc
        self.brain = Brain(brain_cfg.input_dim, 
                           brain_cfg.hidden_dim,
                           brain_cfg.output_dim,
                           brain_cfg.lr)
