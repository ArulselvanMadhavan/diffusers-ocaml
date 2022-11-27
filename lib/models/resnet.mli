open Torch

type resnet_block_2dconfig

val make_config : unit -> resnet_block_2dconfig

type t

val make : Var_store.t -> int -> resnet_block_2dconfig -> t
val print_resnet : t -> unit
val forward : t -> Tensor.t -> Tensor.t option -> Tensor.t
