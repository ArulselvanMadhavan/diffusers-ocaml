open Torch

type resnet_block_2dconfig =
  { out_channels : int option
  ; temb_channels : int option
  ; groups : int
  ; groups_out : int option
  ; eps : float
  ; use_in_shortcut : bool option
  ; output_scale_factor : float
  }

let make () =
  { out_channels = None
  ; temb_channels = Some 512
  ; groups = 32
  ; groups_out = None
  ; eps = 1e-6
  ; use_in_shortcut = None
  ; output_scale_factor = 1.
  }
;;

type t =
  { norm1 : Torch.Nn.t
  ; norm2 : Torch.Nn.t
  ; conv1 : Torch.Nn.t
  ; conv2 : Torch.Nn.t
  ; time_emb_proj : Torch.Nn.t option
  ; conv_shortcut : Torch.Nn.t option
  ; config : resnet_block_2dconfig
  }

    (* pub fn new(vs: nn::Path, in_channels: i64, config: ResnetBlock2DConfig) -> Self { *)

