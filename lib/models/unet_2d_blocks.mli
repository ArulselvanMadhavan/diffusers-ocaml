open Torch

module Downsample2D : sig
  type t

  val make : Var_store.t -> int -> bool -> int -> int -> t
end

module DownEncoderBlock2DConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_downsample : bool
    ; downsample_padding : int
    }

  val default : unit -> t
end

module DownEncoderBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> DownEncoderBlock2DConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module UpDecoderBlock2DConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_upsample : bool
    }

  val default : unit -> t
end

module UpDecoderBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> UpDecoderBlock2DConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module UNetMidBlock2DConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int option
    ; attn_num_head_channels : int option
    ; output_scale_factor : float
    }

  val default : unit -> t
end

module UNetMidBlock2D : sig
  type t

  val make : Var_store.t -> int -> int option -> UNetMidBlock2DConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t
end

module UNetMidBlock2DCrossAttnConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int option
    ; attn_num_head_channels : int
    ; output_scale_factor : float
    ; cross_attn_dim : int
    ; sliced_attention_size : int option
    }

  val default : unit -> t
end

module UNetMidBlock2DCrossAttn : sig
  type t

  val make : Var_store.t -> int -> int option -> UNetMidBlock2DCrossAttnConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t option -> Tensor.t
end

module DownBlock2DConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_downsample : bool
    ; downsample_padding : int
    }

  val default : unit -> t
end

module DownBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int option -> DownBlock2DConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t * Tensor.t list
end

module CrossAttnDownBlock2DConfig : sig
  type t =
    { downblock : DownBlock2DConfig.t
    ; attn_num_head_channels : int
    ; cross_attention_dim : int
    ; sliced_attention_size : int option
    }

  val default : unit -> t
end

module CrossAttnDownBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int option -> CrossAttnDownBlock2DConfig.t -> t

  val forward
    :  t
    -> Tensor.t
    -> Tensor.t option
    -> Tensor.t option
    -> Tensor.t * Tensor.t list
end

module UpBlock2DConfig : sig
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_upsample : bool
    }

  val default : unit -> t
end

module Upsample2D : sig
  type t
end

module UpBlock2D : sig
  type t =
    { resnets : Resnet.ResnetBlock2D.t array
    ; upsampler : Upsample2D.t option
    }

  val make : Var_store.t -> int -> int -> int -> int option -> UpBlock2DConfig.t -> t

  val forward
    :  t
    -> Tensor.t
    -> Tensor.t array
    -> Tensor.t option
    -> (int * int) option
    -> Tensor.t
end

module CrossAttnUpBlock2DConfig : sig
  type t =
    { upblock : UpBlock2DConfig.t
    ; attn_num_head_channels : int
    ; cross_attention_dim : int
    ; sliced_attention_size : int option
    }

  val default : unit -> t
end

module CrossAttnUpBlock2D : sig
  type t =
    { upblock : UpBlock2D.t
    ; attentions : Attention.SpatialTransformer.t list
    }

  val make
    :  Var_store.t
    -> int
    -> int
    -> int
    -> int option
    -> CrossAttnUpBlock2DConfig.t
    -> t

  val forward
    :  t
    -> Tensor.t
    -> Tensor.t array
    -> Tensor.t option
    -> (int * int) option
    -> Tensor.t option
    -> Tensor.t
end
