open Torch

val build_clip_transformer
  :  clip_weights:string
  -> device:Device.t
  -> Diffusers_transformers.Clip.ClipTextTransformer.t

val build_vae
  :  vae_weights:string
  -> device:Device.t
  -> Diffusers_models.Vae.AutoEncoderKL.t

val build_unet
  :  unet_weights:string
  -> device:Device.t
  -> int
  -> int option
  -> Diffusers_models.Unet_2d.UNet2DConditionModel.t
