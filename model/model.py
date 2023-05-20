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


class StyleGan2():
    def __init__(self,
        root,
        num_training_steps=20,
        batch_size=4,
        gradient_accumulate_steps=1,
        num_training_images=-1,
        checkpoint_interval=1000,
        use_loss_regularization=False,
        generate_progress_images=True
    ):
        """
        Entire StyleGAN2 model put together.

        Parameters
        ----------
        root : path to the directory containing the `celeba` folder, either absolute or 
            relative to the script this model is instantiated in. So for example, if this 
            model is instantiated in `test.py`, located inside the root folder of this repo, 
            then `root` should be set to `./`. If this model is instantiated in `training/test.py`,
            `root` should be `../`, and so on.
        num_training_steps : number of steps to train the model. Default value is 20, but should be changed.
        batch_size : batch size to use for training. Default value is 4, but should be changed.
        gradient_accumulate_steps : how many steps to accumulate gradients for before actually updating.
            Default value is 1.
        num_training_images : number (subset) of CelebA images to use for training, or -1 to use
            all 160000+ images. Default value is -1.
        checkpoint_interval : number of steps to wait before saving the next checkpoint and optionally
            generating some output images. Default is 1000.
        use_loss_regularization : whether to use R1 regularization and path length regularization for
            regularizing discriminator and generator losses respectively. Default value is False.
        generate_progress_images : whether to generate a grid of images every `checkpoint_interval` steps.
            Default value is True.
        """
        self.root = root
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.num_training_images = num_training_images
        self.checkpoint_interval = checkpoint_interval
        self.use_regularization = use_loss_regularization
        self.generate_progress_images = generate_progress_images
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters (taken from StyleGAN papers)
        self.dim_latent = 512
        self.adam_betas = (0.0, 0.99)
        self.gan_lr = 0.001  # learning rate for generator and discriminator
        self.mapping_network_lr = self.gan_lr * 0.01  # StyleGAN, Appendix C, 3rd paragraph
        self.gamma = 10  # gradient penalty coefficient gamma


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

            # Add non-default hyperparameters
            "total_num_steps": self.num_training_steps,
            "batch_size": self.batch_size,
            "num_training_images": self.num_training_images,
            "checkpoint_interval": self.checkpoint_interval
        }

        save_path = os.path.join(save_dir, f"stylegan2-{current_num_steps}steps.pth")
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
        seed=None
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

        save_name = f"stylegan2-{current_num_steps}steps-{truncation_psi}trunc"
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
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, self.dim_latent)).type(torch.float32).to(self.device)
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
        Do one step of training the generator (and mapping network).
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
        Do one step.
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
        # Don't think setting .train() makes that much of a difference but it's good practice
        self.mapping_network.train()
        self.generator.train()
        self.discriminator.train()
        LOGGER.info("Starting training")

        avg_gen_loss, avg_disc_loss = 0.0, 0.0

        with tqdm(total=self.num_training_steps, ncols=100) as pbar:
            for step in range(self.num_training_steps):
                # Do one step
                gen_loss, disc_loss = self._do_step(step)
                avg_gen_loss += gen_loss
                avg_disc_loss =+ disc_loss

                pbar.set_postfix(**{
                    "G_loss": gen_loss,
                    "D_loss": disc_loss,
                })
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
                self._save_checkpoint(step+1, avg_gen_loss, avg_disc_loss)
                avg_gen_loss, avg_disc_loss = 0.0, 0.0

                if self.generate_progress_images:
                    LOGGER.info(f"Generating progress images at step {step+1}")
                    self._generate_images(step+1, truncation_psi=0.5)
                    self._generate_images(step+1, truncation_psi=1)
                print()

    
    def generate_output(self,
        num_images,
        num_rows,
        base_path="./",
        truncation_psi=1,
        seed=None
    ):
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
            seed=seed
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
            checkpoint=False
        )


    def load_model(self, path_to_model):
        """
        Load the model from `path_to_model`. The path must contain the <model_name>.pth file itself.
        """
        if not os.path.exists(path_to_model):
            LOGGER.error(f"Model doesn't exist on path {path_to_model}")
            return

        LOGGER.info(f"Loading model from {path_to_model}")

        checkpoint = torch.load(path_to_model, map_location=self.device)

        # Non-default hyperparameters
        self.num_training_steps = checkpoint["total_num_steps"]
        self.batch_size = checkpoint["batch_size"]
        self.num_training_images = checkpoint["num_training_images"]
        self.checkpoint_interval = checkpoint["checkpoint_interval"]
        self.use_regularization = checkpoint["path_length_reg"] is not None
        self.generate_progress_images = True

        # Model
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

        # Optimizers
        self.mapping_network_optim.load_state_dict(checkpoint["mapping_network_optim"])
        self.generator_optim.load_state_dict(checkpoint["generator_optim"])
        self.discriminator_optim.load_state_dict(checkpoint["discriminator_optim"])

        # Reload the data loader
        self.train_loader, _ = setup_data_loaders(self.root, self.batch_size, train_subset_size=self.num_training_images)
        self.train_loader = cycle_data_loader(self.train_loader)
            
        LOGGER.info("Model loaded")