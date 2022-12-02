open Torch

module EncoderConfig : sig
  type t

  val default : unit -> t
end

module DecoderConfig : sig
  type t

  val default : unit -> t
end

module AutoEncoderKLConfig : sig
  type t =
    { block_out_channels : int list
    ; layers_per_block : int
    ; latent_channels : int
    ; norm_num_groups : int
    }

  val default : unit -> t
end

module DiagonalGaussianDistribution : sig
  type t

  val sample : t -> Tensor.t
end

module AutoEncoderKL : sig
  type t

  val make : Var_store.t -> int -> int -> AutoEncoderKLConfig.t -> t
  val encode : t -> Tensor.t -> DiagonalGaussianDistribution.t
  val decode : t -> Tensor.t -> Tensor.t
end
