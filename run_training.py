from model.model import StyleGan2


# Hyperparameters
num_training_steps = 10
batch_size = 4
gradient_accumulate_steps = 4
num_training_images = 20
checkpoint_interval = 3
use_loss_regularization = True
generate_progress_images = True  

# Instantiate
model = StyleGan2(
    root="./",
    num_training_steps=num_training_steps,
    batch_size=batch_size,
    gradient_accumulate_steps=gradient_accumulate_steps,
    num_training_images=num_training_images,
    checkpoint_interval=checkpoint_interval,
    use_loss_regularization=use_loss_regularization,
    generate_progress_images=generate_progress_images
)

# Train and save
model.train_model()
model.save_model()

# Generate some output
# "Bug"?
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/issues/107
# Update: truncation trick might solve it?
# Update 2: that's called mode collapse =) gradient_accumulate_steps might solve it:
# https://github.com/lucidrains/stylegan2-pytorch/issues/183
model.generate_output(16, 4, truncation_psi=0.5)
model.generate_output(16, 4, truncation_psi=0.8)
