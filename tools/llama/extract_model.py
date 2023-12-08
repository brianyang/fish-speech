import torch

state_dict = torch.load(
    "results/text2semantic_400m/checkpoints/step_000025000.ckpt", map_location="cpu"
)["state_dict"]
state_dict = {
    state_dict.replace("model.", ""): value
    for state_dict, value in state_dict.items()
    if state_dict.startswith("model.")
}

torch.save(state_dict, "results/text2semantic_400m/step_000025000_weights.ckpt")
