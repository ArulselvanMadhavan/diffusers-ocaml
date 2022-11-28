open Torch
    
module Downsample2D : sig
  type t

  val make : Var_store.t -> int -> bool -> int -> int -> t
end

module DownEncoderBlock2DConfig : sig
  type t

  val make : unit -> t
end

module DownEncoderBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> DownEncoderBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t
end

module UpDecoderBlock2DConfig : sig
  type t

  val default : unit -> t
end

module UpDecoderBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> UpDecoderBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t
end

module UNetMidBlock2DConfig : sig
  type t

  val default: unit -> t
end

module UNetMidBlock2D : sig
  type t

  val make : Var_store.t -> int -> int option -> UNetMidBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t
end

module UNetMidBlock2DCrossAttnConfig : sig
  type t

  val default : unit -> t
end

module UNetMidBlock2DCrossAttn : sig
  type t

  val make : Var_store.t -> int -> int option -> UNetMidBlock2DCrossAttnConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t option -> Tensor.t
end

module DownBlock2DConfig : sig
  type t

  val default : unit -> t
end

module DownBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int option -> DownBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t * Tensor.t list
end

module CrossAttnDownBlock2DConfig : sig
  type t

  val default : unit -> t
    
end

module CrossAttnDownBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int option -> CrossAttnDownBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t option -> Tensor.t * Tensor.t list
end

module UpBlock2DConfig : sig
  type t

  val default : unit -> t
end

module UpBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int -> int option -> UpBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t array -> Tensor.t option -> (int * int) option -> Tensor.t
end

module CrossAttnUpBlock2DConfig : sig
  type t

  val default : unit -> t
end

module CrossAttnUpBlock2D : sig
  type t

  val make : Var_store.t -> int -> int -> int -> int option -> CrossAttnUpBlock2DConfig.t -> t

  val forward : t -> Tensor.t -> Tensor.t array -> Tensor.t option -> (int * int) option -> Tensor.t option -> Tensor.t
end

















