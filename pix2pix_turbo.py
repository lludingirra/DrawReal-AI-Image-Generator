import os
import requests
import sys
import copy
from tqdm import tqdm # Progress bar library
import torch # PyTorch deep learning framework
from transformers import AutoTokenizer, CLIPTextModel # For text encoding
from diffusers import AutoencoderKL, UNet2DConditionModel # Core diffusion model components
from diffusers.utils.peft_utils import set_weights_and_activate_adapters # For managing PEFT adapters
from peft import LoraConfig # For LoRA (Low-Rank Adaptation) configuration

# Add 'src/' to the system path if models are in a subfolder.
# Assuming 'model.py' is directly in the same directory as 'pix2pix_turbo.py',
# this line might be redundant or indicative of a larger project structure.
p = "src/"
sys.path.append(p)

# Import custom functions from 'model.py'
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    """
    A custom convolution module designed to blend outputs from a pre-trained
    convolution layer and a newly trained/modified convolution layer.
    The blending ratio 'r' controls the contribution of each.
    """
    def __init__(self, convin_pretrained, convin_curr):
        """
        Initializes the TwinConv module.
        
        Args:
            convin_pretrained (torch.nn.Module): A deep copy of the pre-trained
                                                 convolution layer. Its weights
                                                 are detached during forward pass.
            convin_curr (torch.nn.Module): The current (potentially trained/LoRA-adapted)
                                           convolution layer.
        """
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None # Blending factor, typically set externally (e.g., in forward pass)

    def forward(self, x):
        """
        Performs the forward pass by blending the outputs of the two convolution layers.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Blended output tensor.
        """
        # Get output from the pre-trained convolution, detaching it to prevent gradient flow.
        x1 = self.conv_in_pretrained(x).detach()
        # Get output from the current convolution.
        x2 = self.conv_in_curr(x)
        # Blend the outputs based on the 'r' factor: r * x2 + (1 - r) * x1.
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    """
    A wrapper class for the Pix2Pix-Turbo model, integrating components like
    CLIP text encoder, VAE, and UNet from Diffusers, along with custom
    modifications for fast image-to-image translation (e.g., single-step denoising)
    and LoRA adaptation.
    """
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        """
        Initializes the Pix2Pix_Turbo model.
        
        Args:
            pretrained_name (str, optional): Name of a predefined pre-trained model
                                             ("edge_to_image" or "sketch_to_image_stochastic").
                                             If provided, model weights are downloaded.
            pretrained_path (str, optional): Path to a local pre-trained model checkpoint (.pkl).
                                             If provided, weights are loaded from this path.
            ckpt_folder (str): Directory to store downloaded model checkpoints.
            lora_rank_unet (int): Rank for LoRA adaptation in the UNet.
            lora_rank_vae (int): Rank for LoRA adaptation in the VAE.
        """
        super().__init__()
        # Load CLIP tokenizer and text encoder for conditioning.
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        # Initialize a 1-step noise scheduler for fast inference.
        self.sched = make_1step_sched()

        # Load the VAE (Variational AutoEncoder) component.
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        # Replace the VAE encoder/decoder forward passes with custom versions
        # that support skip connections.
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        
        # Add custom skip connection convolution layers to the VAE decoder.
        # These are used to incorporate feature maps from the encoder into the decoder.
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False # Enable skip connections

        # Load the UNet (U-shaped Network) component.
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # --- Conditional Model Loading Based on Pre-trained Configuration ---
        if pretrained_name == "edge_to_image":
            # Define URL for edge-to-image specific LoRA weights.
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True) # Create checkpoint folder if it doesn't exist.
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl") # Define local save path.
            
            # Download checkpoint if it does not exist.
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong during download: Incomplete file.")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf # Set checkpoint path.
            
            # Load the state dictionary from the checkpoint.
            sd = torch.load(p_ckpt, map_location="cpu")
            # Configure LoRA for UNet and VAE based on loaded ranks and target modules.
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            
            # Add LoRA adapter to VAE and load its weights.
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            
            # Add LoRA adapter to UNet and load its weights.
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # Similar logic for "sketch_to_image_stochastic" model.
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            
            # Download checkpoint if it does not exist.
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong during download: Incomplete file.")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            
            # For stochastic models, replace UNet's initial convolution with TwinConv
            # to blend between a pre-trained and a LoRA-adapted conv.
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            
            # Load state dictionary and configure LoRA for UNet and VAE.
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            # Load model from a local path if specified.
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            # Initialize model with random weights and apply LoRA configurations.
            print("Initializing model with random weights")
            # Initialize skip connection convolutions with small constant values.
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            
            # Define target modules for VAE LoRA.
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                  "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                  "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                         target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            
            # Define target modules for UNet LoRA.
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                                          target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            
            # Store LoRA ranks and target modules for saving.
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention() # Optional: Uncomment for memory efficiency with xformers
        unet.to("cuda") # Move UNet to CUDA device
        vae.to("cuda") # Move VAE to CUDA device
        self.unet, self.vae = unet, vae # Store UNet and VAE as instance attributes
        self.vae.decoder.gamma = 1 # Initialize gamma for VAE decoder (used in blending skip connections)
        self.timesteps = torch.tensor([999], device="cuda").long() # Set timestep for single-step inference
        self.text_encoder.requires_grad_(False) # Freeze text encoder weights

    def set_eval(self):
        """
        Sets the model components (UNet and VAE) to evaluation mode and
        disables gradient calculation for all their parameters.
        """
        self.unet.eval() # Set UNet to evaluation mode
        self.vae.eval() # Set VAE to evaluation mode
        self.unet.requires_grad_(False) # Disable gradient tracking for UNet
        self.vae.requires_grad_(False) # Disable gradient tracking for VAE

    def set_train(self):
        """
        Sets the model components (UNet and VAE) to training mode and
        enables gradient calculation specifically for LoRA parameters
        and custom convolution layers.
        """
        self.unet.train() # Set UNet to training mode
        self.vae.train() # Set VAE to training mode
        
        # Enable gradients only for LoRA layers within UNet.
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        # Also enable gradients for the custom conv_in layer in UNet (if TwinConv is used).
        self.unet.conv_in.requires_grad_(True)
        
        # Enable gradients only for LoRA layers within VAE.
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        # Enable gradients for custom skip connection convolution layers in VAE decoder.
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        """
        Performs the forward pass for image generation.
        
        Args:
            c_t (torch.Tensor): Conditional input image tensor (e.g., edge map, sketch).
            prompt (str, optional): Text prompt for conditioning.
            prompt_tokens (torch.Tensor, optional): Pre-encoded text tokens.
            deterministic (bool): If True, uses deterministic sampling. If False,
                                  uses a stochastic approach with 'r' blending.
            r (float): Blending factor for stochastic mode, controlling the mix
                       between conditional input and noise, and LoRA weight scaling.
            noise_map (torch.Tensor, optional): Pre-generated noise map for stochastic mode.
            
        Returns:
            torch.Tensor: The generated output image tensor.
        """
        # Ensure either prompt or prompt_tokens is provided, but not both.
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        # Encode the text prompt if provided.
        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0] # Use provided tokens if available

        if deterministic:
            # --- Deterministic Inference Path ---
            # Encode the conditional input (e.g., Canny edges) to latent space.
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # UNet predicts the noise based on encoded control and text conditioning.
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            # Scheduler steps to denoise the latent representation.
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype) # Ensure consistent dtype
            
            # Pass encoder's intermediate activations to the decoder for skip connections.
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            # Decode the denoised latent representation back to pixel space.
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # --- Stochastic Inference Path (Controlled by 'r' blending factor) ---
            # Set LoRA adapter weights for UNet based on 'r' (blending factor).
            self.unet.set_adapters(["default"], weights=[r])
            # Set LoRA adapter weights for VAE skip connections based on 'r'.
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            
            # Encode conditional input.
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # Blend the encoded control input with noise based on 'r'.
            unet_input = encoded_control * r + noise_map * (1 - r)
            
            # Set the blending factor for TwinConv (if used in UNet's conv_in).
            self.unet.conv_in.r = r
            # UNet predicts noise.
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None # Reset blending factor after use
            
            # Denoising step.
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            
            # Pass encoder's intermediate activations and gamma for decoder blending.
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            # Decode the denoised latent representation.
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        """
        Saves the relevant parts of the model's state dictionary (LoRA weights and custom layers)
        to a specified output file.
        
        Args:
            outf (str): Output file path (.pkl) for saving the model state.
        """
        sd = {} # Initialize an empty dictionary for saving state.
        # Store LoRA target modules and ranks for both UNet and VAE.
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        
        # Store state dictionaries for LoRA weights and specific custom layers
        # (e.g., conv_in for UNet, skip convolutions for VAE decoder).
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        
        torch.save(sd, outf) # Save the state dictionary to the file.
