"""
Modified NCSNpp Generator with Four-Parameter Conditioning and NaN Handling
This implementation extends the original NCSNpp model to support conditioning on four parameters
with proper NaN handling (especially setting 4th parameter to 0 if NaN).
"""

import torch.nn as nn
import functools
import torch
import numpy as np
from . import utils, layers, layerspp, dense_layer

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense

def safe_parameter_handling_generator(param):
    """
    Safe parameter handling for generator with NaN checking
    Specifically handles 4th parameter NaN -> 0.0 conversion
    """
    if param is None:
        return None
    
    # Ensure it's a tensor
    if not isinstance(param, torch.Tensor):
        param = torch.tensor(param, dtype=torch.float32)
    
    # Check for NaN values and replace with zeros (especially 4th param)
    nan_mask = torch.isnan(param)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum().item()} NaN values found in generator parameters")
        param = param.clone()
        
        # Special handling for 4th parameter
        if param.dim() == 2 and param.shape[1] >= 4:
            fourth_param_nans = nan_mask[:, 3]
            if fourth_param_nans.any():
                print(f"Warning: {fourth_param_nans.sum().item()} NaN values in 4th parameter, setting to 0.0")
                param[fourth_param_nans, 3] = 0.0
        
        # Handle other NaN values
        param[nan_mask] = 0.0
    
    return param

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

# Modified NCSNpp model with 4-parameter conditioning
@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model with four-parameter conditioning and NaN handling"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.not_use_tanh = config.not_use_tanh
    self.act = act = nn.SiLU()
    self.z_emb_dim = z_emb_dim = config.z_emb_dim
    self.param_emb_dim = config.param_emb_dim if hasattr(config, 'param_emb_dim') else self.z_emb_dim
    
    # Parameter transformation for 4 parameters
    self.param_transform = nn.Sequential(
        nn.Linear(4, self.param_emb_dim // 2),  # Changed from 3 to 4 input dimensions
        self.act,
        nn.Linear(self.param_emb_dim // 2, self.param_emb_dim)
    )
    
    self.nf = nf = config.num_channels_dae
    ch_mult = config.ch_mult
    self.num_res_blocks = num_res_blocks = config.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.attn_resolutions
    dropout = config.dropout
    resamp_with_conv = config.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.conditional  # noise-conditional
    fir = config.fir
    fir_kernel = config.fir_kernel
    self.skip_rescale = skip_rescale = config.skip_rescale
    self.resblock_type = resblock_type = config.resblock_type.lower()
    self.progressive = progressive = config.progressive.lower()
    self.progressive_input = progressive_input = config.progressive_input.lower()
    self.embedding_type = embedding_type = config.embedding_type.lower()
    init_scale = 0.
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)
    elif resblock_type == 'biggan_oneadagn':
      ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = config.num_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)
    
    mapping_layers = [PixelNorm(),
                      dense(config.nz, z_emb_dim),
                      self.act,]
    for _ in range(config.n_mlp):
        mapping_layers.append(dense(z_emb_dim, z_emb_dim))
        mapping_layers.append(self.act)
    self.z_transform = nn.Sequential(*mapping_layers)
    

  def forward(self, x, time_cond, z, param=None):
    # timestep/noise_level embedding; only for continuous training
    zemb = self.z_transform(z)
    modules = self.all_modules
    m_idx = 0
    
    pemb = None
    if param is not None:
        # Apply safe parameter handling with NaN checking
        param = safe_parameter_handling_generator(param)
        
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
                elif param.size(0) != x.size(0):
                    # DDP scatters positional args but not kwargs, so param retains the
                    # full batch size while x has already been split per GPU.
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        rank = dist.get_rank()
                        local_bs = x.size(0)
                        param = param[rank * local_bs:(rank + 1) * local_bs]
                    else:
                        param = param[:x.size(0)]
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
        param = safe_parameter_handling_generator(param)
        
        # Create parameter embedding
        pemb = self.param_transform(param.float())
    
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
     
      temb = layers.get_timestep_embedding(timesteps, self.nf)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      if pemb is not None:
        pemb = modules[m_idx-1](pemb)
      if pemb is not None:
            temb = temb + pemb 
      
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if not self.config.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb, zemb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb, zemb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb, zemb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb, zemb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb, zemb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    
    if not self.not_use_tanh:
        return torch.tanh(h)
    else:
        return h

# Test function for parameter validation in generator
def test_generator_parameter_handling():
    """
    Test function to validate parameter handling and NaN conversion in generator
    """
    print("Testing generator parameter handling...")
    
    # Test cases
    test_cases = [
        [0.1, 0.2, 0.3, 0.4],  # Normal case
        [0.1, 0.2, 0.3, float('nan')],  # NaN in 4th parameter
        [0.1, float('nan'), 0.3, 0.4],  # NaN in middle
        [float('nan'), float('nan'), float('nan'), float('nan')],  # All NaN
        [0.5, 0.6, 0.7],  # Only 3 parameters (should pad to 4)
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 5 parameters (should truncate to 4)
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        
        # Convert to tensor
        param_tensor = torch.tensor([test_case], dtype=torch.float32)
        print(f"Before processing: {param_tensor}")
        
        # Apply safe handling
        processed = safe_parameter_handling_generator(param_tensor)
        print(f"After processing: {processed}")
        
        # Check if NaN values were properly handled
        has_nan = torch.isnan(processed).any()
        print(f"Contains NaN after processing: {has_nan}")
        print(f"Shape: {processed.shape}")
        
        if not has_nan and processed.shape[1] == 4:
            print("✓ Test passed - no NaN values remain and correct shape")
        else:
            print("✗ Test failed - issues remain")

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_generator_parameter_handling()