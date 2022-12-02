open Torch

module BlockConfig : sig
  type t =
    { out_channels : int
    ; use_cross_attn : bool
    }
end

module UNet2DConditionModelConfig : sig
  type t =
    { center_input_sample : bool
    ; flip_sin_to_cos : bool
    ; freq_shift : float
    ; blocks : BlockConfig.t array
    ; layers_per_block : int
    ; downsample_padding : int
    ; mid_block_scale_factor : float
    ; norm_num_groups : int
    ; norm_eps : float
    ; cross_attention_dim : int
    ; attention_head_dim : int
    ; sliced_attention_size : int option
    }

  val default : unit -> t
end

module UNet2DConditionModel : sig
  type t

  val make : Var_store.t -> int -> int -> UNet2DConditionModelConfig.t -> t
  val forward : t -> Tensor.t -> float -> Tensor.t -> Tensor.t
end
