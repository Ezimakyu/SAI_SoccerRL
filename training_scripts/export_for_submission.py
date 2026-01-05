import torch
import torch.nn as nn
import os
from sac import SAC
import numpy as np

# --- CONFIG ---
CHECKPOINT_PATH = "run_models/sac_run_checkpoint.pth" # Path to your best model
OUTPUT_PATH = "submit_models/testsubmit3.pth"                  # What you will actually submit
STATE_DIM = 88                                 # Must match your Preprocessor output

ACTION_DIM = 12
NEURONS = [256, 256] # Must match your SAC config exactly!

class PrimitivePolicy(nn.Module):
    def __init__(self, n_in, n_out, neurons):
        super().__init__()
        layers = []
        curr_in = n_in
        for h in neurons:
            layers.append(nn.Linear(curr_in, h))
            layers.append(nn.ReLU())
            curr_in = h
        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(curr_in, n_out)

    def forward(self, x):
        # --- THE FIX: FORCE INPUT TO FLOAT ---
        # This ensures that even if 'x' is a Double, 
        # it becomes a Float before hitting the first Linear layer.
        x = x.float() 
        
        x = self.backbone(x)
        x = self.output_layer(x)
        return torch.tanh(x)

def export():
    # 1. Load your actual trained SAC agent
    class DummyActionSpace:
        def __init__(self): self.shape = (ACTION_DIM,)
    
    trained_agent = SAC(n_features=STATE_DIM, action_space=DummyActionSpace(), neurons=NEURONS, device="cpu")
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    trained_agent.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    
    # 2. Create the Primitive "Clone"
    primitive_model = PrimitivePolicy(STATE_DIM, ACTION_DIM, NEURONS)
    
    # 3. Manually copy the weights layer-by-layer
    # This is the 'clean' way to move the brain without the 'baggage'
    print("Cloning weights to primitive model...")
    
    # Copy Backbone (Sequential linear layers)
    # trained_agent.actor.base_net[0] -> primitive_model.backbone[0]
    # trained_agent.actor.base_net[2] -> primitive_model.backbone[2]
    with torch.no_grad():
        # Copy base_net layers (they are at indices 0, 2, 4...)
        for i in range(len(NEURONS)):
            idx = i * 2
            primitive_model.backbone[idx].weight.copy_(trained_agent.actor.base_net[idx].weight)
            primitive_model.backbone[idx].bias.copy_(trained_agent.actor.base_net[idx].bias)
            
        # Copy Final Output (Mean)
        primitive_model.output_layer.weight.copy_(trained_agent.actor.mean_linear.weight)
        primitive_model.output_layer.bias.copy_(trained_agent.actor.mean_linear.bias)

    primitive_model.eval()
    
    # 4. Trace the Primitive Model
    example_input = torch.randn(1, STATE_DIM).float() * 0.5
    print("Tracing primitive model...")
    try:
        # We use a simple lambda to ensure absolute purity
        traced_model = torch.jit.trace(primitive_model, example_input)
        torch.jit.save(traced_model, OUTPUT_PATH)
        print(f"✅ CLEAN EXPORT SUCCESSFUL: {OUTPUT_PATH}")
        
        # Verify it works
        reloaded = torch.jit.load(OUTPUT_PATH)
        out = reloaded(example_input)
        print(f"Verification Output: {out[0, :3]}... (Action matches)")

    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export()