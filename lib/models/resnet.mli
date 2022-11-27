open Torch

module ResnetBlock2DConfig : sig
  type t =
    { out_channels : int option
    ; temb_channels : int option
    ; groups : int
    ; groups_out : int option
    ; eps : float
    ; use_in_shortcut : bool option
    ; output_scale_factor : float
    }

  val default : unit -> t
end

module ResnetBlock2D : sig
  type t

  val make : Var_store.t -> int -> ResnetBlock2DConfig.t -> t
  val print_resnet : t -> unit
  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t
end
