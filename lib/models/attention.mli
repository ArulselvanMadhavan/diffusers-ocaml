open Torch

module SpatialTransformerConfig : sig
  type t

  val make : unit -> t
end

module SpatialTransformer : sig
  type t

  val make : Var_store.t -> int -> int -> int -> SpatialTransformerConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t
end

module AttentionBlockConfig : sig
  type t

  val make : unit -> t
end

module AttentionBlock : sig
  type t

  val make : Var_store.t -> int -> AttentionBlockConfig.t -> t
  val forward : t -> Tensor.t -> Tensor.t
end
