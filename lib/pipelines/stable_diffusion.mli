open Torch

val build_clip_transformer
  :  clip_weights:string
  -> device:Device.t
  -> Diffusers_transformers.Clip.ClipTextTransformer.t
