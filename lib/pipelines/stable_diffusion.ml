open Torch

let build_clip_transformer ~clip_weights ~device =
  let vs = Var_store.create ~device ~name:"" () in
  let text_model = Diffusers_transformers.Clip.ClipTextTransformer.make vs in
  let named_tensors = Var_store.all_vars vs in
  Serialize.load_multi_ ~named_tensors ~filename:clip_weights;
  text_model
;;

let build_vae ~vae_weights ~device =
  let vs = Var_store.create ~device ~name:"" () in
  let autoencoder_cfg =
    Diffusers_models.Vae.AutoEncoderKLConfig.
      { block_out_channels = [ 128; 256; 512; 512 ]
      ; layers_per_block = 2
      ; latent_channels = 4
      ; norm_num_groups = 32
      }
  in
  let autoencoder = Diffusers_models.Vae.AutoEncoderKL.make vs 3 3 autoencoder_cfg in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:vae_weights;
  autoencoder
;;

let build_unet ~unet_weights ~device in_channels sliced_attention_size =
  let open Diffusers_models.Unet_2d in
  let vs = Var_store.create ~device ~name:"" () in
  let blocks =
    [| BlockConfig.{ out_channels = 320; use_cross_attn = true }
     ; BlockConfig.{ out_channels = 640; use_cross_attn = true }
     ; BlockConfig.{ out_channels = 1280; use_cross_attn = true }
     ; BlockConfig.{ out_channels = 1280; use_cross_attn = false }
    |]
  in
  let unet_cfg =
    UNet2DConditionModelConfig.
      { attention_head_dim = 8
      ; blocks
      ; center_input_sample = false
      ; cross_attention_dim = 768
      ; downsample_padding = 1
      ; flip_sin_to_cos = true
      ; freq_shift = 0.
      ; layers_per_block = 2
      ; mid_block_scale_factor = 1.
      ; norm_eps = 1e-5
      ; norm_num_groups = 32
      ; sliced_attention_size
      }
  in
  let unet = UNet2DConditionModel.make vs in_channels 4 unet_cfg in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:unet_weights;
  unet
;;
