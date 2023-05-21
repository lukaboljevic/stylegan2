from model.model import StyleGan2


"""
Script to train model from scratch
"""

# Hyperparameters
root                        = "./"          # Directory where `celeba` is located
model_idx                   = 1             # Unique identifier for the model, mostly for saving purposes
gan_lr                      = 0.002         # Generator and discriminator LR
mapping_network_lr          = gan_lr * 0.1  # Mapping network LR
num_training_images         = 70000         # Number of CelebA training images
num_training_steps          = 10000         # Number of training steps
batch_size                  = 32            # Batch size
dim_latent                  = 512           # Dimensionality of latent variables `z` and `w`
adam_betas                  = (0.0, 0.99)   # Betas for Adam optimizer
gamma                       = 10            # Gradient penalty coefficient gamma
gradient_accumulate_steps   = 4             # How many steps to accumulate gradients for
use_loss_regularization     = True          # Use R1 (gradient penalty) and path length regularization
checkpoint_interval         = 1000          # How often to save a checkpoint
generate_progress_images    = True          # Whether to also generate some images every `checkpoint_interval` steps

# Instantiate
model = StyleGan2(
    root=root,
    model_index=model_idx,
    gan_lr=gan_lr,
    mapping_network_lr=mapping_network_lr,
    num_training_images=num_training_images,
    num_training_steps=num_training_steps,
    batch_size=batch_size,
    dim_latent=dim_latent,
    adam_betas=adam_betas,
    gamma=gamma,
    gradient_accumulate_steps=gradient_accumulate_steps,
    use_loss_regularization=use_loss_regularization,
    checkpoint_interval=checkpoint_interval,
    generate_progress_images=generate_progress_images,
)

# Train and save
model.train_model()
model.save_model()

# Generate some output
# WARNING: If all 16 images are pretty much exactly the same, even after ~1k steps,
# you are experiencing mode collapse.
# Try different learning rates, and/or different gradient_accumulate_steps. Those are
# generally the culprits (at least from my experience)
model.generate_output(16, 4, truncation_psi=0.5)
model.generate_output(16, 4, truncation_psi=0.8)
