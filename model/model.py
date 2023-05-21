import json
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from general_utils.generator_noise import generate_noise
from general_utils.logger import setup_logger
from general_utils.losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    PathLengthRegularization,
    GradientPenalty,
)

from .dataset import setup_data_loaders, cycle_data_loader
from .mapping_network import MappingNetwork
from .generator import Generator
from .discriminator import Discriminator


LOGGER = setup_logger(__name__)


class StyleGan2:
    def __init__(self,
        root,
        model_index=1,
        gan_lr=0.002,
        mapping_network_lr=0.0002,
        num_training_images=50000,
        num_training_steps=5000,
        batch_size=32,
        dim_latent=512,
        adam_betas=(0.0, 0.99),
        gamma=10,
        gradient_accumulate_steps=1,
        use_loss_regularization=False,
        checkpoint_interval=1000,
        generate_progress_images=True,
    ):
        """
        Entire StyleGAN2 model put together.

        Parameters
        ----------
        `root` : Path to the directory containing the `celeba` folder, either absolute or
            relative to the script this model is instantiated in.
        `model_index` : Integer that uniquely identifies the model, mainly used for saving checkpoints.
            Default value is 0, but should be changed.
        `gan_lr` : Generator and discriminator learning rate. Default value is 0.001.
        `mapping_network_lr` : Mapping network learning rate. Default value is 0.0001.
        `num_training_images` : How many CelebA images to use for training, or -1 to use all 160000+ images.
            Default value is -1.
        `num_training_steps` : Number of steps to train the model. Default value is 20, but should be changed.
        `batch_size` : batch Size to use for training. Default value is 4, but should be changed.
        `dim_latent` : Dimensionality of latent variables `z` and `w`, Default value is 512.
        `adam_betas` : Tuple containing beta_1 and beta_2 for Adam optimizer. Default value is (0.0, 0.99)
        `gamma` : Gradient penalty coefficient gamma. Default value is 10.
        `gradient_accumulate_steps` : How many steps to accumulate gradients for before actually updating.
            Default value is 1.
        `use_loss_regularization` : Whether to use R1 regularization and path length regularization for
            regularizing discriminator and generator losses respectively. Default value is False.
        `checkpoint_interval` : Number of steps to wait before saving the next checkpoint and optionally
            generating some output images. Default value is 1000.
        `generate_progress_images` : Whether to generate a grid of images every `checkpoint_interval` steps.
            Default value is True.
        """
        self.model_index = model_index
        self.root = root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.gan_lr = gan_lr
        self.mapping_network_lr = mapping_network_lr
        self.num_training_images = num_training_images
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        self.adam_betas = adam_betas
        self.gamma = gamma
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.use_regularization = use_loss_regularization

        # For saving checkpoints and progress images
        self.checkpoint_interval = checkpoint_interval
        self.generate_progress_images = generate_progress_images

        # Dataset
        self.train_loader, _ = setup_data_loaders(self.root, self.batch_size, train_subset_size=self.num_training_images)
        self.train_loader = cycle_data_loader(self.train_loader)

        # StyleGAN2
        self.mapping_network = MappingNetwork().to(self.device)
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Loss functions and their (optional) regularizations
        self.generator_loss_fn = GeneratorLoss().to(self.device)
        self.discriminator_loss_fn = DiscriminatorLoss().to(self.device)
        self.path_length_reg = None
        self.gradient_penalty = None

        if self.use_regularization:
            self.path_length_reg = PathLengthRegularization().to(self.device)
            self.gradient_penalty = GradientPenalty().to(self.device)
            # Lazy regularization parameters (StyleGAN2):
            # "... The regularization terms can be computed less frequently than the main loss functions ..."
            # "... R1 regularization is performed only once every 16 minibatches ..."
            self.gradient_penalty_interval = 16
            self.path_length_interval = 16

        # Optimizers
        self.generator_optim = optim.Adam(
            self.generator.parameters(),
            lr=self.gan_lr,
            betas=self.adam_betas
        )
        self.discriminator_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.gan_lr,
            betas=self.adam_betas
        )
        self.mapping_network_optim = optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_lr,
            betas=self.adam_betas
        )

        # Put everything together to log when we start training
        self.hyperparameters = {
            "Num train steps": self.num_training_steps,
            "Num train images": self.num_training_images,
            "Gen and Disc LR": self.gan_lr,
            "Mapping LR": self.mapping_network_lr,
            "Batch size": self.batch_size,
            "Latent dim": self.dim_latent,
            "Adam betas": self.adam_betas,
            "R1 reg. gamma": self.gamma,
            "Gradient accumulate steps": self.gradient_accumulate_steps,
            "Use loss regularization": self.use_regularization,
            "Save checkpoint every": self.checkpoint_interval,
        }

        LOGGER.info("Model set up")


    def _save_checkpoint(self,
        current_num_steps,
        avg_generator_loss,
        avg_discriminator_loss,
        base_path=os.getcwd(),
        checkpoint=True
    ):
        """
        Save a general checkpoint, using `base_path` as the root directory. If `checkpoint=True`,
        saves the model in `base_path/checkpoints` directory, otherwise it is saved in `base_path`.
        """
        if checkpoint:
            save_dir = os.path.join(base_path, "checkpoints")
        else:
            save_dir = base_path

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dict = {
            # Add current progress
            "current_num_steps": current_num_steps,
            "mapping_network": self.mapping_network.state_dict(),
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "path_length_reg": self.path_length_reg.state_dict() if self.path_length_reg is not None else None,
            "mapping_network_optim": self.mapping_network_optim.state_dict(),
            "generator_optim": self.generator_optim.state_dict(),
            "discriminator_optim": self.discriminator_optim.state_dict(),
            "avg_generator_loss": avg_generator_loss,
            "avg_discriminator_loss": avg_discriminator_loss,
            # Add hyperparameters and stuff
            "model_index": self.model_index,
            "gan_lr": self.gan_lr,
            "mapping_network_lr": self.mapping_network_lr,
            "num_training_images": self.num_training_images,
            "num_training_steps": self.num_training_steps,
            "batch_size": self.batch_size,
            "dim_latent": self.dim_latent,
            "adam_betas": self.adam_betas,
            "gamma": self.gamma,
            "gradient_accumulate_steps": self.gradient_accumulate_steps,
            "use_regularization": self.use_regularization,
            "checkpoint_interval": self.checkpoint_interval,
        }

        save_path = os.path.join(save_dir, f"stylegan2-{self.model_index}idx-{current_num_steps}steps.pth")
        LOGGER.info(f"Saving model after {current_num_steps} steps")
        torch.save(save_dict, save_path)
        LOGGER.info(f"Model saved to {save_path}")


    def _generate_images(self,
        current_num_steps,
        num_images=16,
        num_rows=4,
        base_path=os.getcwd(),
        checkpoint=True,
        truncation_psi=1,
        seed=None,
    ):
        """
        Generate images using the current state of generator, using `base_path` as the root directory.
        If `checkpoint=True`, save to `base_path/checkpoints`, otherwise save to `base_path`.
        """
        if checkpoint:
            save_dir = os.path.join(base_path, "checkpoints")
        else:
            save_dir = base_path

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_name = f"stylegan2-{self.model_index}idx-{current_num_steps}steps-{truncation_psi}trunc"
        if seed is not None:
            save_name += f"-{seed}seed"
        save_name += ".png"
        save_path = os.path.join(save_dir, save_name)

        with torch.no_grad():
            images, _ = self._generator_output(num_images, truncation_psi=truncation_psi, seed=seed)

            # Old way
            # images = torch.clamp(images, 0, 1)
            # image_grid = make_grid(images, nrow=num_rows, padding=0).permute(1, 2, 0).cpu().numpy()
            # image_grid = Image.fromarray(np.uint8(image_grid*255)).convert("RGB")
            # image_grid.save(save_path)

            # Idea taken from NVIDIA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/generate.py#L116
            images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_grid = make_grid(images, nrow=num_rows).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(image_grid, mode="RGB").save(save_path)

        LOGGER.info(f"Generated images saved to {save_path}")


    def _generator_output(self, batch_size=-1, truncation_psi=1, seed=None):
        """
        Use the generator to generate `batch_size` images. Default value for `batch_size`
        is -1, meaning that self.batch_size is used in its place.

        The function returns batch_size generated images and the intermediate latent
        variable `w` that was one of the inputs to the generator.
        """
        if batch_size == -1:
            batch_size = self.batch_size

        # Generate images
        if seed is None:
            z = torch.randn(batch_size, self.dim_latent).to(self.device)
        else:
            z = (
                torch.from_numpy(np.random.RandomState(seed).randn(batch_size, self.dim_latent))
                .type(torch.float32)
                .to(self.device)
            )
        w = self.mapping_network(z, truncation_psi=truncation_psi)
        noise = generate_noise(batch_size, self.device)
        generated_images = self.generator(w, noise)

        return generated_images, w


    def _discriminator_step(self, inputs, step_idx):
        """
        Do one step of training the discriminator.
        """
        self.discriminator_optim.zero_grad()
        total_discriminator_loss = torch.tensor(0.0).to(self.device)

        # Accumulate gradient for certain number of steps
        for _ in range(self.gradient_accumulate_steps):
            # Generate images
            generated_images, _ = self._generator_output()

            # Discriminator classification for generated images
            fake_output = self.discriminator(generated_images.detach())

            # Real images
            real_images = inputs.to(self.device)
            if self.use_regularization and step_idx % self.gradient_penalty_interval == 0:
                real_images.requires_grad_()

            # Discriminator classification for real images
            real_output = self.discriminator(real_images)

            # Get discriminator loss
            real_loss, fake_loss = self.discriminator_loss_fn(real_output, fake_output)
            discriminator_loss = real_loss + fake_loss

            # Add gradient penalty once very self.gradient_penalty_interval steps
            if self.use_regularization and step_idx % self.gradient_penalty_interval == 0:
                grad_penalty = self.gradient_penalty(real_images, real_output)
                discriminator_loss = discriminator_loss + 0.5 * self.gamma * grad_penalty

            # Normalize/scale
            total_discriminator_loss += discriminator_loss.detach().item() / self.gradient_accumulate_steps
            discriminator_loss = discriminator_loss / self.gradient_accumulate_steps

            # Calculate gradients
            discriminator_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        # Take optimizer step
        self.discriminator_optim.step()

        return total_discriminator_loss


    def _generator_step(self, step_idx):
        """
        Do one step of training the generator and mapping network.
        """
        self.generator_optim.zero_grad()
        self.mapping_network_optim.zero_grad()
        total_generator_loss = torch.tensor(0.0).to(self.device)

        # Accumulate gradient for certain number of steps
        for _ in range(self.gradient_accumulate_steps):
            # Generate images
            generated_images, w = self._generator_output()

            # Discriminator classification for generated images
            fake_output = self.discriminator(generated_images)

            # Get generator loss
            generator_loss = self.generator_loss_fn(fake_output)

            # Calculate path length penalty once very self.path_length_interval steps
            if self.use_regularization and step_idx % self.path_length_interval == 0:
                path_length_penalty = self.path_length_reg(w, generated_images)
                generator_loss = generator_loss + path_length_penalty

            # Normalize/scale
            total_generator_loss += generator_loss.detach().item() / self.gradient_accumulate_steps
            generator_loss = generator_loss / self.gradient_accumulate_steps

            # Calculate gradients
            generator_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

        # Take optimizer step
        self.generator_optim.step()
        self.mapping_network_optim.step()

        return total_generator_loss


    def _do_step(self, step_idx):
        """
        Do one training step.
        """
        # We don't care about the targets
        batch_num, (inputs, _) = next(self.train_loader)

        # Train the discriminator first for a step
        discriminator_loss = self._discriminator_step(inputs, step_idx)

        # Train the generator (and mapping network) next
        generator_loss = self._generator_step(step_idx)

        # Return just the values of losses
        return generator_loss.item(), discriminator_loss.item()


    def train_model(self):
        """
        Train the model. The function prints the hyperparameters that are used at the beginning.
        """
        # Don't think setting .train() makes that much of a difference but it's good practice
        self.mapping_network.train()
        self.generator.train()
        self.discriminator.train()
        LOGGER.info("Starting training")
        LOGGER.info(f"Model index: {self.model_index}")
        LOGGER.info(f"Training on {self.device.upper()}")
        LOGGER.info(f"Setup:\n{json.dumps(self.hyperparameters, indent=4)}")

        avg_gen_loss, avg_disc_loss = 0.0, 0.0

        with tqdm(total=self.num_training_steps, ncols=100) as pbar:
            for step in range(self.num_training_steps):
                # Do one step
                gen_loss, disc_loss = self._do_step(step)
                avg_gen_loss += gen_loss
                avg_disc_loss += disc_loss

                pbar.set_postfix(
                    **{
                        "G_loss": gen_loss,
                        "D_loss": disc_loss,
                    }
                )
                pbar.update()

                if (step + 1) % self.checkpoint_interval != 0:
                    continue

                # Every checkpoint_interval steps, save checkpoint and optionally generate output images
                print()
                avg_gen_loss /= self.checkpoint_interval
                avg_disc_loss /= self.checkpoint_interval
                LOGGER.info(f"Average GEN loss after {step+1} steps: {avg_gen_loss}")
                LOGGER.info(f"Average DISC loss after {step+1} steps: {avg_disc_loss}")

                # Save current average losses for when we invoke .save_model()
                self.final_avg_gen_loss = avg_gen_loss
                self.final_avg_disc_loss = avg_disc_loss

                # Save checkpoint and reset average losses
                self._save_checkpoint(step + 1, avg_gen_loss, avg_disc_loss)
                avg_gen_loss, avg_disc_loss = 0.0, 0.0

                if self.generate_progress_images:
                    LOGGER.info(f"Generating progress images at step {step+1}")
                    self._generate_images(step + 1, truncation_psi=0.5)
                    self._generate_images(step + 1, truncation_psi=1)
                print()


    def generate_output(self, num_images, num_rows, base_path="./", truncation_psi=1, seed=None):
        """
        Generate `num_images` images in a grid with `num_rows` rows using the fully
        trained generator. Images are saved in `base_path`.
        """
        LOGGER.info(f"Generating images")
        self._generate_images(
            self.num_training_steps,
            num_images=num_images,
            num_rows=num_rows,
            base_path=base_path,
            checkpoint=False,
            truncation_psi=truncation_psi,
            seed=seed,
        )


    def save_model(self, base_path="./"):
        """
        Save the final trained model to `base_path`.
        """
        self._save_checkpoint(
            self.num_training_steps,
            self.final_avg_gen_loss,
            self.final_avg_disc_loss,
            base_path=base_path,
            checkpoint=False,
        )


    def load_model(self, path_to_model: str):
        """
        Load the model from `path_to_model`. The path must contain the <model_name>.pth file itself.
        """
        if not os.path.exists(path_to_model):
            LOGGER.error(f"Model doesn't exist on path {path_to_model}")
            return

        if not path_to_model.endswith((".pth", ".pt")):
            LOGGER.error(f"The path to model doesn't end with '.pt' or '.pth'.")
            return

        LOGGER.info(f"Loading model from {path_to_model}")

        checkpoint = torch.load(path_to_model, map_location=self.device)

        # Read all hyperparameters
        self.model_index = checkpoint["model_index"]
        self.gan_lr = checkpoint["gan_lr"]
        self.mapping_network_lr = checkpoint["mapping_network_lr"]
        self.num_training_images = checkpoint["num_training_images"]
        self.num_training_steps = checkpoint["num_training_steps"]
        self.batch_size = checkpoint["batch_size"]
        self.dim_latent = checkpoint["dim_latent"]
        self.adam_betas = tuple(checkpoint["adam_betas"])
        self.gamma = checkpoint["gamma"]
        self.gradient_accumulate_steps = checkpoint["gradient_accumulate_steps"]
        self.use_regularization = checkpoint["use_regularization"]
        self.checkpoint_interval = checkpoint["checkpoint_interval"]
        self.generate_progress_images = True

        # Load model state dicts
        self.mapping_network.load_state_dict(checkpoint["mapping_network"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])

        # Regularization was optional
        if self.use_regularization:
            self.path_length_reg = PathLengthRegularization().to(self.device)
            self.gradient_penalty = GradientPenalty().to(self.device)
            self.gradient_penalty_interval = 16
            self.path_length_interval = 16
            self.path_length_reg.load_state_dict(checkpoint["path_length_reg"])

        # Load optimizer state dicts
        self.mapping_network_optim.load_state_dict(checkpoint["mapping_network_optim"])
        self.generator_optim.load_state_dict(checkpoint["generator_optim"])
        self.discriminator_optim.load_state_dict(checkpoint["discriminator_optim"])

        # Reload the data loader
        self.train_loader, _ = setup_data_loaders(
            self.root, self.batch_size, train_subset_size=self.num_training_images
        )
        self.train_loader = cycle_data_loader(self.train_loader)

        # Put everything together to log when we start training
        self.hyperparameters = {
            "Num train steps": self.num_training_steps,
            "Num train images": self.num_training_images,
            "Gen and Disc LR": self.gan_lr,
            "Mapping LR": self.mapping_network_lr,
            "Batch size": self.batch_size,
            "Latent dim": self.dim_latent,
            "Adam betas": self.adam_betas,
            "R1 reg. gamma": self.gamma,
            "Gradient accumulate steps": self.gradient_accumulate_steps,
            "Use loss regularization": self.use_regularization,
            "Save checkpoint every": self.checkpoint_interval,
        }

        LOGGER.info("Model loaded")
