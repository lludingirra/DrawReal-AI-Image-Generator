import os
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler


def make_1step_sched():
    """
    Creates and configures a 1-step DDPM noise scheduler for fast inference.
    This scheduler is initialized from a pre-trained StabilityAI SD-Turbo model
    and set to perform denoising in a single step.

    Returns:
        DDPMScheduler: Configured DDPM noise scheduler.
    """
    # Load the DDPMScheduler from the pre-trained "stabilityai/sd-turbo" model,
    # specifically from its "scheduler" subfolder.
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    # Set the number of inference timesteps to 1 for a single-step generation.
    # Move the scheduler to the CUDA device if available for faster computation.
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    # Ensure the cumulative product of alphas (used in denoising) is also on the CUDA device.
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    """
    Custom forward pass for the VAE encoder. This function mimics the original
    VAE encoder's behavior but also stores intermediate down-block outputs
    (skip connections) which can be used by a modified decoder.

    Args:
        self (AutoencoderKL): The VAE encoder instance.
        sample (torch.Tensor): The input tensor to the VAE encoder.

    Returns:
        torch.Tensor: The output of the VAE encoder (latent representation).
    """
    # Apply the initial convolution layer.
    sample = self.conv_in(sample)
    l_blocks = [] # List to store outputs from down-blocks (for skip connections).

    # Process through the VAE encoder's down-sampling blocks.
    for down_block in self.down_blocks:
        l_blocks.append(sample) # Store the input to the current down-block.
        sample = down_block(sample) # Apply the down-sampling block.

    # Process through the VAE encoder's middle block.
    sample = self.mid_block(sample)
    # Apply the final normalization, activation, and convolution layers.
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    # Store the collected down-block outputs for later use by the decoder.
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    """
    Custom forward pass for the VAE decoder, supporting skip connections
    from the encoder and a gamma parameter for blending. This modified decoder
    can leverage feature maps from the encoder to improve reconstruction quality,
    especially when 'ignore_skip' is False.

    Args:
        self (AutoencoderKL): The VAE decoder instance.
        sample (torch.Tensor): The input latent tensor to the VAE decoder.
        latent_embeds (torch.Tensor, optional): Optional latent embeddings to condition the mid-block.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """
    # Apply the initial convolution layer for the decoder.
    sample = self.conv_in(sample)
    # Determine the upscale data type based on the first up-block's parameters.
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    # Process through the VAE decoder's middle block.
    sample = self.mid_block(sample, latent_embeds)
    # Cast the sample to the appropriate upscale data type.
    sample = sample.to(upscale_dtype)

    # Check if skip connections should be used.
    if not self.ignore_skip:
        # Define the skip convolution layers.
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # Iterate through the up-sampling blocks.
        for idx, up_block in enumerate(self.up_blocks):
            # Calculate the skip connection input: reverse the incoming skip acts,
            # apply the corresponding skip convolution, and scale by gamma.
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # Add the skip connection to the current sample.
            sample = sample + skip_in
            # Apply the up-sampling block.
            sample = up_block(sample, latent_embeds)
    else:
        # If skip connections are ignored, simply apply the up-sampling blocks.
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    # Apply post-processing layers (normalization, activation, final convolution).
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds) # Apply conditioning if latent_embeds present
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    """
    Downloads a file from a given URL to a specified output path with a progress bar.
    If the file already exists at the output path, the download is skipped.

    Args:
        url (str): The URL of the file to download.
        outf (str): The local path where the file should be saved.
    """
    # Check if the output file already exists.
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        # Send a GET request to the URL, enabling streaming to handle large files.
        response = requests.get(url, stream=True)
        # Get the total size of the file from the response headers; default to 0 if not available.
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # Define chunk size for downloading (1 Kibibyte).
        # Initialize a TQDM progress bar to visualize the download progress.
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        # Open the output file in binary write mode.
        with open(outf, 'wb') as file:
            # Iterate over content in chunks and update the progress bar.
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data) # Write the downloaded chunk to the file.
        progress_bar.close() # Close the progress bar upon completion.
        # Verify if the download was complete by comparing downloaded bytes with total size.
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download: Incomplete file.")
        print(f"Downloaded successfully to {outf}")
    else:
        # If the file already exists, skip the download and inform the user.
        print(f"Skipping download, {outf} already exists")

