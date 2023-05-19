import torch

from model.model import StyleGan2


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1
batch_size = 4
num_training_images = 8
save_every_num_epoch = 1
use_loss_regularization = True
generate_progress_images = True  

# Instantiate
model = StyleGan2(
    root="./",
    num_epochs=num_epochs,
    batch_size=batch_size,
    num_training_images=num_training_images,
    save_every_num_epoch=save_every_num_epoch,
    use_loss_regularization=use_loss_regularization,
    generate_progress_images=generate_progress_images
)

# Train and save
model.train_model()
model.save_model()

# Generate some output
# TODO there's still a bug with generation ... 
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/issues/107
# Update: truncation trick might solve it?
model.generate_output(16, 4, truncation_psi=0.5)
model.generate_output(16, 4, truncation_psi=0.8)
