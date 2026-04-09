import torch
import torch.nn as nn
import torch.optim as optim

class Brain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train_step(self, input, target):
        self.train()
        
        state_tensor = torch.tensor(input, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        predictions = self.forward(state_tensor)
        loss = self.criterion(predictions, target_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), predictions.detach().numpy()

# input_size = 10   
# hidden_size = 24  
# output_size = 4   

# brain = Brain(input_size, hidden_size, output_size)

# # --- Simulation Loop ---

# fixed_state = [0.5, 0.1, -0.2, 0.8, 0.0, 1.0, -0.5, 0.3, 0.2, 0.9]
# target_phase_1 = [1.0, 0.0, 0.5, -1.0]
# target_phase_2 = [-0.5, 0.8, -0.2, 0.5] 
# epochs = 200

# def np_round(arr):
#     return [round(x, 4) for x in arr.tolist()]

# print("Starting Simulation...\n")

# for epoch in range(epochs):
#     current_target = target_phase_1 if epoch < 100 else target_phase_2
    
#     loss_val, current_preds = brain.train_step(fixed_state, current_target)
    
#     if epoch == 0:
#         print(f"--- INITIAL STATE ---")
#         print(f"Target:     {current_target}")
#         print(f"Prediction: {np_round(current_preds)}\n")

#     elif epoch == 99:
#         print(f"--- END OF PHASE 1 ---")
#         print(f"Target:     {current_target}")
#         print(f"Prediction: {np_round(current_preds)}")
#         print(f"Loss:       {loss_val:.6f}\n")
        
#     elif epoch == 100:
#         print(f"--- START OF PHASE 2 ---")
#         print(f"Target:     {current_target}")
#         print(f"Prediction: {np_round(current_preds)}")
#         print(f"Loss Spikes to: {loss_val:.6f}\n")

#     elif epoch == 199:
#         print(f"--- END OF PHASE 2 ---")
#         print(f"Target:     {current_target}")
#         print(f"Prediction: {np_round(current_preds)}")
#         print(f"Loss:       {loss_val:.6f}\n")
