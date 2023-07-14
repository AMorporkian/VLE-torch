import cv2
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from PIL import Image
from prodigyopt import Prodigy
import wandb
from VLE import VLE


num_epochs = 5
max_tokens = 15
batch_size = 8


# The 'get_loaders' function loads the ImageNet dataset, filters out images smaller than 256x256,
# and returns a PyTorch DataLoader for the first 10000 images
def get_loaders():
    dataset = load_dataset("imagenet-1k")
    train_dataset = (
        dataset["train"]
        #.select(range(10000))
        #.filter(lambda x: x["image"].width >= 256 and x["image"].height >= 256)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return train_loader

# The 'norm' and 'tot' functions are used to normalize and convert images to tensors
tot = torchvision.transforms.PILToTensor()
norm = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)
# The 'collate_fn' function is used to collate the images into a batch
def collate_fn(batch):
    images = [x["image"].convert("RGB") for x in batch]
    images = [
        x.resize(
            (256, 256),
        )
        for x in images
    ]
    images = [norm(tot(x).to(dtype=torch.float)) for x in images]
    images = torch.stack(images)
    return images

# Initialize the Accelerator with mixed precision and Weights & Biases logging
accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")
accelerator.project_configuration.automatic_checkpoint_naming = True
accelerator.init_trackers("VLE practice")

# Initialize the model and optimizer
vle = VLE()
optimizer = Prodigy(vle.parameters(), growth_rate=1.02, d_coef=0.1)

# Get the DataLoader
train_loader = get_loaders()

# Move the models and the optimizer to the appropriate device
train_loader, vle, optimizer = accelerator.prepare(train_loader, vle, optimizer)


# The 'VLETrainer' class encapsulates the training process for the VLE model
class VLETrainer:
    def __init__(
        self, model, optimizer, train_loader, accelerator, num_epochs, max_tokens
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.num_epochs = num_epochs
        self.max_tokens = max_tokens
        self.steps = 1
    # The 'train' method trains the model for the specified number of epochs
    def train(self):
        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                loss = self.train_step(batch)
                if self.steps % 10 == 0 or self.steps == 1:
                    self.create_and_log_images(
                        self.steps,
                        batch,
                        self.current_mask,
                        self.current_reconstruction,
                        loss,
                    )
                else:
                    self.log_loss(self.steps, loss)
                self.steps += 1
            if self.steps % 10000 == 0:
                self.accelerator.save_state(f"checkpoints")    
            self.accelerator.save_state("checkpoints/")

    # The 'train_step' method performs one step of training on a given batch of data
    def train_step(self, batch):
        self.optimizer.zero_grad()
        batch = batch.to(self.accelerator.device)

        self.current_reconstruction = torch.zeros_like(batch)
        self.current_mask = torch.zeros((batch.shape[0], 1, 256, 256), device=batch.device)

        loss_recreation = torch.tensor(0.0, device=self.accelerator.device)
        loss_mask = torch.tensor(0.0, device=self.accelerator.device)
        token_count = self.get_max_tokens(self.steps)
        for _ in range(token_count):
            residual = batch - self.current_reconstruction
            partial_reconstruction, mask = self.model(residual)

            self.current_reconstruction = (
                self.current_reconstruction + partial_reconstruction
            )
            self.current_mask = self.current_mask + mask

            loss_recreation = loss_recreation + torch.linalg.vector_norm(
                mask.mul(residual - partial_reconstruction)
            ) / torch.numel(partial_reconstruction)
            loss_mask = loss_mask + torch.exp(-F.mse_loss(mask, self.current_mask))

            # Free the memory
            del residual, partial_reconstruction, mask

        loss = (loss_recreation + loss_mask) / token_count
        loss = loss.to(self.accelerator.device)
        self.accelerator.backward(loss)
        self.optimizer.step()

        # Free the memory
        del batch
        torch.cuda.empty_cache()

        return loss
    def get_max_tokens(self, steps):
                # Define the standard deviation of the distribution
        std_dev = 1.0

        # Define the initial mean of the distribution
        initial_mean = 1

        # Define the rate at which the mean increases
        mean_increase_rate = 0.0001

    
        # Increase the mean linearly with the number of iterations
        mean = initial_mean + mean_increase_rate * self.steps
        num_steps = int(round(abs(np.random.normal(mean, std_dev))))
        num_steps = min(num_steps, self.max_tokens)
        num_steps = max(num_steps, 2)
        return num_steps
    # The 'create_and_log_images' method creates images and logs them to Weights & Biases
    def create_and_log_images(self, step, inputs, mask, reconstruction, loss):
        # Convert tensors to images
        image, mask, reconstruction = self.convert_image_from_tensor(
            inputs[0], mask[0], reconstruction[0]
        )
        # Create heatmap
        heatmap = self.create_heatmap(mask, image)

        # Create a matplotlib figure
        fig, ax = plt.subplots(1, 7, figsize=(35, 5))

        # Plot the four images
        ax[0].imshow(image)
        ax[0].set_title("Original")

        # Plot mask channels separately
        ax[1].imshow(mask[:,:,0], cmap="gray")
        ax[1].set_title("Mask Channel 0 (Red)")
        ax[2].imshow(mask[:,:,1], cmap="gray")
        ax[2].set_title("Mask Channel 1 (Green)")
        ax[3].imshow(mask[:,:,2], cmap="gray")
        ax[3].set_title("Mask Channel 2 (Blue)")

        ax[4].imshow(reconstruction)
        ax[4].set_title("Reconstruction")
        ax[5].imshow(heatmap)
        ax[5].set_title("Heatmap")

        # Remove the axis
        for a in ax:
            a.axis("off")

        # Log the figure with wandb
        self.accelerator.log({"images": wandb.Image(fig), "loss": loss})
        # Close the figure
        plt.close(fig)
        
    # The 'log_loss' method logs the loss to Weights & Biases
    def log_loss(self, step, loss):
        self.accelerator.print(f"Step{step}, Loss {loss.item()}")
        self.accelerator.log({"loss": loss})

    # The 'create_heatmap' method creates a heatmap from a mask and an image
    def create_heatmap(self, mask: np.ndarray, image: np.ndarray):
        # First, resize the mask to the same size as the image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        # Create a heatmap by applying a colormap to the mask
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_CIVIDIS)
        # Blend the heatmap and the image together
        heatmap = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        # Create a PIL image from the heatmap
        heatmap = Image.fromarray(heatmap)
        return heatmap

    def convert_image_from_tensor(self, *images):
        i = []
        for image in images:
            # Convert tensor to a numpy array, denormalize the image
            image = image.detach().cpu()
            if image.shape[0] == 1:
                # Check if it only has one channel, if it does, convert it to 3 channels
                image = image.repeat(3, 1, 1)
            image = image.permute(1, 2, 0)
            # Compress to the range [0, 1]
            image = image.numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            i.append(image)
        return i
# Load checkpoint
#accelerator.load_state("checkpoints/")

# Then, to train the model, you can use:
trainer = VLETrainer(vle, optimizer, train_loader, accelerator, num_epochs, max_tokens)
trainer.train()
