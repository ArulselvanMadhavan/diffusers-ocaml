open Torch

module TimestepEmbedding : sig
  type t

  val make : Var_store.t -> int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module Timesteps : sig
  type t

  val make : int -> bool -> float -> Device.t -> t
  val forward : t -> Tensor.t -> Tensor.t
end
