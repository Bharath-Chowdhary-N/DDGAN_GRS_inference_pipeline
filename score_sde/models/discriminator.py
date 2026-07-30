"""
Modified Discriminator for Four-Parameter Conditioning with NaN Handling
This file updates the discriminator implementation to support four parameters with proper NaN handling.
"""

import torch
import torch.nn as nn
import numpy as np

from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

def safe_parameter_handling_discriminator(param):
    """
    Safe parameter handling for discriminator with NaN checking
    """
    if param is None:
        return None
    
    # Ensure it's a tensor
    if not isinstance(param, torch.Tensor):
        param = torch.tensor(param, dtype=torch.float32)
    
    # Check for NaN values and replace with zeros (especially 4th param)
    nan_mask = torch.isnan(param)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum().item()} NaN values found in discriminator parameters")
        param = param.clone()
        param[nan_mask] = 0.0
        
        # Special handling for 4th parameter
        if param.dim() == 2 and param.shape[1] >= 4:
            fourth_param_nans = nan_mask[:, 3]
            if fourth_param_nans.any():
                print(f"Warning: {fourth_param_nans.sum().item()} NaN values in 4th parameter, setting to 0.0")
    
    return param

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding),
                    )

        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.)
                    )
        self.dense_t1= dense(t_emb_dim, out_channel)

        self.act = act
        
        self.skip = nn.Sequential(
                    conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                    )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
       
        out = self.act(out)
       
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out
    
class Discriminator_small(nn.Module):
  """A time-dependent discriminator for small images (CIFAR10, StackMNIST) with 4-parameter conditioning."""

  def __init__(self, nc = 3, ngf = 64, t_emb_dim = 128, act=nn.LeakyReLU(0.2), param_emb_dim=128):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    # Parameter embedding for 4 parameters
    self.param_embedding = nn.Sequential(
            nn.Linear(4, param_emb_dim // 2),  # Input: 4 parameters
            nn.SiLU(),
            nn.Linear(param_emb_dim // 2, param_emb_dim)
        )
    
    self.t_embed = TimestepEmbedding(
        embedding_dim=t_emb_dim,
        hidden_dim=t_emb_dim,
        output_dim=t_emb_dim,
        act=act,
        )
     
    # Encoding layers where the resolution decreases
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim = t_emb_dim,act=act)
    
    self.conv2 = DownConvBlock(ngf*2, ngf*4,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    self.conv3 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)

    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1)
    
    self.stddev_group = 4
    self.stddev_feat = 1
        
  def forward(self, x, t, x_t, param=None):
    t_embed = self.act(self.t_embed(t))
    
    # Add parameter embedding if provided
    if param is not None:
        # Apply safe parameter handling with NaN checking
        param = safe_parameter_handling_discriminator(param)
        
        # Handle 4 parameters
        if isinstance(param, (list, tuple)) and len(param) == 4:
            # Convert list/tuple of 4 values to tensor
            param = torch.tensor([param], device=x.device, dtype=torch.float32).repeat(x.shape[0], 1)
        elif isinstance(param, torch.Tensor):
            if param.dim() == 1 and param.size(0) == 4:
                # Single set of 4 parameters
                param = param.unsqueeze(0).repeat(x.shape[0], 1)
            elif param.dim() == 2 and param.size(1) == 4:
                # Batch of parameters - make sure batch size matches
                if param.size(0) == 1 and x.size(0) > 1:
                    param = param.repeat(x.shape[0], 1)
            elif param.dim() == 2 and param.size(1) < 4:
                # Pad with zeros if fewer than 4 parameters
                padding = torch.zeros(param.size(0), 4 - param.size(1), 
                                    device=param.device, dtype=param.dtype)
                param = torch.cat([param, padding], dim=1)
            elif param.dim() == 2 and param.size(1) > 4:
                # Truncate to 4 parameters
                param = param[:, :4]
            else:
                raise ValueError(f"Parameter tensor must have shape [batch_size, 4], got {param.shape}")
        else:
            raise ValueError("Parameter must be a tensor of shape [batch_size, 4] or a list/tuple of 4 values")
            
        # Final safety check for NaN values after processing
        param = safe_parameter_handling_discriminator(param)
        
        param_embed = self.param_embedding(param.float())
        # Combine with time embedding
        t_embed = t_embed + param_embed
    
    input_x = torch.cat((x, x_t), dim = 1)
    
    h0 = self.start_conv(input_x)
    h1 = self.conv1(h0,t_embed)    
    
    h2 = self.conv2(h1,t_embed)   
   
    h3 = self.conv3(h2,t_embed)
   
    out = self.conv4(h3,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
   
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out


class Discriminator_large(nn.Module):
  """A time-dependent discriminator for large images (CelebA, LSUN) with 4-parameter conditioning."""

  def __init__(self, nc = 1, ngf = 32, t_emb_dim = 128, param_emb_dim=128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
      
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*4, t_emb_dim = t_emb_dim, downsample = True, act=act)
    
    self.conv2 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)

    self.conv3 = DownConvBlock(ngf*8, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    # Parameter embedding for 4 parameters
    self.param_embedding = nn.Sequential(
        nn.Linear(4, param_emb_dim // 2),  # Input: 4 parameters
        nn.SiLU(),
        nn.Linear(param_emb_dim // 2, param_emb_dim)
    )
    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.conv5 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.conv6 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)

    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1)
    self.end_linear = dense(ngf*8, 1)
    
    self.stddev_group = 4
    self.stddev_feat = 1
        
  def forward(self, x, t, x_t, param=None):
    t_embed = self.act(self.t_embed(t))  
    
    if param is not None:
        # Apply safe parameter handling with NaN checking
        param = safe_parameter_handling_discriminator(param)
        
        # Handle 4 parameters
        if isinstance(param, (list, tuple)) and len(param) == 4:
            # Convert list/tuple of 4 values to tensor
            param = torch.tensor([param], device=x.device, dtype=torch.float32).repeat(x.shape[0], 1)
        elif isinstance(param, torch.Tensor):
            if param.dim() == 1 and param.size(0) == 4:
                # Single set of 4 parameters
                param = param.unsqueeze(0).repeat(x.shape[0], 1)
            elif param.dim() == 2 and param.size(1) == 4:
                # Batch of parameters - make sure batch size matches
                if param.size(0) == 1 and x.size(0) > 1:
                    param = param.repeat(x.shape[0], 1)
            elif param.dim() == 2 and param.size(1) < 4:
                # Pad with zeros if fewer than 4 parameters
                padding = torch.zeros(param.size(0), 4 - param.size(1), 
                                    device=param.device, dtype=param.dtype)
                param = torch.cat([param, padding], dim=1)
            elif param.dim() == 2 and param.size(1) > 4:
                # Truncate to 4 parameters
                param = param[:, :4]
            else:
                raise ValueError(f"Parameter tensor must have shape [batch_size, 4], got {param.shape}")
        else:
            raise ValueError("Parameter must be a tensor of shape [batch_size, 4] or a list/tuple of 4 values")
            
        # Final safety check for NaN values after processing
        param = safe_parameter_handling_discriminator(param)
        
        param_embed = self.param_embedding(param.float())
        # Combine with time embedding
        t_embed = t_embed + param_embed
    
    input_x = torch.cat((x, x_t), dim = 1)
    
    h = self.start_conv(input_x)
    h = self.conv1(h,t_embed)    
   
    h = self.conv2(h,t_embed)
   
    h = self.conv3(h,t_embed)
    h = self.conv4(h,t_embed)
    h = self.conv5(h,t_embed)
   
    out = self.conv6(h,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out

# Create a config file to help with testing conditional generation
def create_sample_config(output_file, param_values=None):
    """
    Create a sample configuration file for testing conditional generation with four parameters
    """
    import json
    
    if param_values is None:
        # Default to a few example parameter sets (now with 4 parameters)
        param_values = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.4, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.0],  # All minimum
            [1.0, 1.0, 1.0, 1.0],  # All maximum
            [0.5, 0.5, 0.5, 0.5],  # All middle
        ]
    
    config = {
        "samples": [
            {"param_values": values, "num_images": 4} 
            for values in param_values
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created sample config at {output_file}")

# Test function for parameter validation
def test_parameter_handling():
    """
    Test function to validate parameter handling and NaN conversion
    """
    print("Testing parameter handling...")
    
    # Test cases
    test_cases = [
        [0.1, 0.2, 0.3, 0.4],  # Normal case
        [0.1, 0.2, 0.3, float('nan')],  # NaN in 4th parameter
        [0.1, float('nan'), 0.3, 0.4],  # NaN in middle
        [float('nan'), float('nan'), float('nan'), float('nan')],  # All NaN
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        
        # Convert to tensor
        param_tensor = torch.tensor([test_case], dtype=torch.float32)
        print(f"Before processing: {param_tensor}")
        
        # Apply safe handling
        processed = safe_parameter_handling_discriminator(param_tensor)
        print(f"After processing: {processed}")
        
        # Check if NaN values were properly handled
        has_nan = torch.isnan(processed).any()
        print(f"Contains NaN after processing: {has_nan}")
        
        if not has_nan:
            print("✓ Test passed - no NaN values remain")
        else:
            print("✗ Test failed - NaN values still present")

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_parameter_handling()
    create_sample_config("sample_config_4params.json")