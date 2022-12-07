open Torch

val cpu_or_cuda : string list -> string -> Device.t
val gen_tokens : string -> Device.t -> Tensor.t * Tensor.t
val build_text_embeddings : string -> Device.t -> Tensor.t -> Tensor.t -> Tensor.t
