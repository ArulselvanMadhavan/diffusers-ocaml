open Torch
open Diffusers_models
open Diffusers_schedulers

val cpu_or_cuda : string list -> string -> Device.t
val gen_tokens : string -> Device.t -> Tensor.t * Tensor.t
val build_text_embeddings : string -> Device.t -> Tensor.t -> Tensor.t -> Tensor.t

val update_latents
  :  Tensor.t
  -> Unet_2d.UNet2DConditionModel.t
  -> int
  -> Tensor.t
  -> Ddim.DDimScheduler.t
  -> Tensor.t

val build_image
  :  int
  -> int
  -> Device.t
  -> Vae.AutoEncoderKL.t
  -> Tensor.t
  -> string
  -> unit
