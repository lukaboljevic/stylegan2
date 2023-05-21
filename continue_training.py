from model.model import StyleGan2


"""
Load a pretrained model and train it for an additional X steps
"""

root = "./"  # Directory where `celeba` is located
model = StyleGan2(root=root)

# Pretrained model
model.load_model(path_to_model="./stylegan2-3idx-10000steps.pth")

# Train for an additional 10 steps
model.train_model(continue_training_for=10)

# Save
model.save_model()