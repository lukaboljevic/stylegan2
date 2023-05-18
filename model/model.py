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

from .dataset import setup_data_loaders
from .mapping_network import MappingNetwork
from .generator import Generator
from .discriminator import Discriminator


LOGGER = setup_logger(__name__)


class StyleGan2():
    def __init__(self,
        root,
        num_epochs,
        batch_size,
        num_training_images=-1,
        save_every_num_epoch=-1,
        use_loss_regularization=False,
        generate_progress_images=True
    ):
        """
        Parameters
        ----------
        root : path to the directory containing the `celeba` folder, either absolute or 
            relative to the script this model is instantiated in. So for example, if this 
            model is instantiated in `test.py`, located inside the root folder of this repo, 
            then `root` should be set to `./`. If this model is instantiated in `training/test.py`,
            `root` should be `../`, and so on.
        num_epochs : number of epochs to train the model
        batch_size : batch size to use for training
        num_training_images : number (subset) of CelebA images to use for training, or -1 to use
            all 160000+ images. Default value is -1.
        save_every_num_epoch : number of epochs to wait before saving the next checkpoint, or -1
            to not save any checkpoints. Default value is -1.
        use_loss_regularization : whether to use GradientPenalty and PathLengthRegularization modules
            for regularizing discriminator and generator losses respectively. Default value is False.
        generate_progress_images : whether to generate a grid of images at the end of each epoch to
            show current training progress. Default value is True.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_training_images = num_training_images
        self.save_every_num_epoch = save_every_num_epoch
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
        self.train_loader, _ = setup_data_loaders(root, self.batch_size, train_subset_size=self.num_training_images)

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


    def _discriminator_step(self, inputs, batch_num):
        """
        Do one step of training the discriminator.
        """
        self.discriminator_optim.zero_grad()

        # Generate images
        generated_images, _ = self._generator_output()

        # Discriminator classification for generated images
        fake_output = self.discriminator(generated_images.detach())

        # Get real images from the data loader
        real_images = inputs.to(self.device)
        if self.use_regularization and batch_num % self.gradient_penalty_interval == 0:
            real_images.requires_grad_()

        # Discriminator classification for real images
        real_output = self.discriminator(real_images)

        # Get discriminator loss
        real_loss, fake_loss = self.discriminator_loss_fn(real_output, fake_output)
        discriminator_loss = real_loss + fake_loss

        # Add gradient penalty once very self.gradient_penalty_interval minibatches
        if self.use_regularization and batch_num % self.gradient_penalty_interval == 0:
            grad_penalty = self.gradient_penalty(real_images, real_output)
            discriminator_loss = discriminator_loss + 0.5 * self.gamma * grad_penalty

        # Calculate gradients
        discriminator_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        # Take optimizer step
        self.discriminator_optim.step()

        return discriminator_loss
    

    def _generator_step(self, batch_num):
        """
        Do one step of training the generator (and mapping network).
        """
        self.generator_optim.zero_grad()
        self.mapping_network_optim.zero_grad()

        # Generate images
        generated_images, w = self._generator_output()

        # Discriminator classification for generated images
        fake_output = self.discriminator(generated_images)

        # Get generator loss
        generator_loss = self.generator_loss_fn(fake_output)

        # Calculate path length penalty once very self.path_length_interval minibatches
        if self.use_regularization and batch_num % self.path_length_interval == 0:
            path_length_penalty = self.path_length_reg(w, generated_images)
            generator_loss = generator_loss + path_length_penalty

        # Calculate gradients
        generator_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

        # Take optimizer step
        self.generator_optim.step()
        self.mapping_network_optim.step()

        return generator_loss


    def _do_epoch(self, progress_bar):
        """
        Do one epoch.
        """
        avg_generator_loss = 0.0
        avg_discriminator_loss = 0.0

        for batch_num, data in enumerate(self.train_loader, start=1):
            inputs, _ = data  # we don't care about the targets

            # Train the discriminator first
            discriminator_loss = self._discriminator_step(inputs, batch_num)

            # Train the generator (and mapping network) next
            generator_loss = self._generator_step(batch_num)

            # Update progress bar 'n' stuff
            avg_generator_loss += generator_loss.item()
            avg_discriminator_loss += discriminator_loss.item()
            progress_bar.set_postfix(**{
                "D_loss": discriminator_loss.item(),
                "G_loss": generator_loss.item()
            })
            progress_bar.update(self.batch_size)
        
        # Calculate average losses
        avg_generator_loss /= len(self.train_loader)*1.0
        avg_discriminator_loss /= len(self.train_loader)*1.0

        return avg_generator_loss, avg_discriminator_loss


    def _save_checkpoint(self,
        current_num_epochs,
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
            "current_num_epochs": current_num_epochs,
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
            "total_num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "num_training_images": self.num_training_images,
            "save_every_num_epoch": self.save_every_num_epoch,
        }

        save_path = os.path.join(save_dir, f"stylegan2-{current_num_epochs}epochs.pth")
        LOGGER.info(f"Saving model after {current_num_epochs} epochs to {save_path}")
        torch.save(save_dict, save_path)
        LOGGER.info("Model saved")


    def _generate_images(self,
        current_num_epochs,
        num_images=16,
        num_rows=4,
        base_path=os.getcwd(),
        checkpoint=True
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
        save_path = os.path.join(save_dir, f"stylegan2-{current_num_epochs}epochs.png")

        LOGGER.info(f"Generating progress images after {current_num_epochs} epochs")
        with torch.no_grad():
            images, _ = self._generator_output(num_images)
            images = torch.clamp(images, 0, 1)
            image_grid = make_grid(images, nrow=num_rows, padding=0).permute(1, 2, 0).cpu().numpy()
            image_grid = Image.fromarray(np.uint8(image_grid*255)).convert("RGB")
            image_grid.save(save_path)
        LOGGER.info(f"Generated images saved to {save_path}")


    def _generator_output(self, batch_size=-1):
        """
        Use the generator to generate `batch_size` images. Default value for `batch_size`
        is -1, meaning that self.batch_size is used in its place.

        The function returns batch_size generated images and the intermediate latent 
        variable `w` that was one of the inputs to the generator.
        """
        if batch_size == -1:
            batch_size = self.batch_size

        # Generate images
        z = torch.randn(batch_size, self.dim_latent).to(self.device)
        w = self.mapping_network(z)
        noise = generate_noise(batch_size, self.device)
        generated_images = self.generator(w, noise)

        return generated_images, w


    def save_model(self, base_path="./"):
        """
        Save the final trained model to `base_path`.
        """
        self._save_checkpoint(
            self.num_epochs,
            self.final_avg_gen_loss,
            self.final_avg_disc_loss,
            base_path=base_path,
            checkpoint=False
        )


    def train_model(self):
        # Don't think it makes that much of a difference but it's good practice
        self.mapping_network.train()
        self.generator.train()
        self.discriminator.train()
        save_counter = 0
        LOGGER.info("Starting training")

        for epoch in range(self.num_epochs):
            print()
            with tqdm(
                total=len(self.train_loader.dataset),
                desc=f"Epoch: {epoch+1}/{self.num_epochs}",
                unit="images",
                ncols=100
            ) as progress_bar:
                avg_gen_loss, avg_disc_loss = self._do_epoch(progress_bar)
                print()
                LOGGER.info(f"Average GEN loss: {avg_gen_loss}")
                LOGGER.info(f"Average DISC loss: {avg_disc_loss}")

                if self.generate_progress_images:
                    self._generate_images(epoch+1)

                save_counter += 1
                if self.save_every_num_epoch == -1:
                    # Don't save any checkpoints
                    continue

                if save_counter == self.save_every_num_epoch:
                    # Save a checkpoint every save_every_num_epoch epochs
                    save_counter = 0
                    self._save_checkpoint(epoch+1, avg_gen_loss, avg_disc_loss)

        # Store the final loss values so we can save the entire model easily
        self.final_avg_gen_loss = avg_gen_loss
        self.final_avg_disc_loss = avg_disc_loss

    
    def generate_output(self, num_images, num_rows, base_path="./"):
        """
        Generate `num_images` images in a grid with `num_rows` rows using the fully
        trained generator. Images are saved in `base_path`.
        """
        self._generate_images(
            self.num_epochs,
            num_images=num_images,
            num_rows=num_rows,
            base_path=base_path,
            checkpoint=False
        )