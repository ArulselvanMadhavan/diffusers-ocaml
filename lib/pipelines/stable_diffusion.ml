open Torch

let build_clip_transformer ~clip_weights ~device =
  let vs = Var_store.create ~device ~name:"" () in
  let text_model = Diffusers_transformers.Clip.ClipTextTransformer.make vs in
  let named_tensors = Var_store.all_vars vs in
  Serialize.load_multi_ ~named_tensors ~filename:clip_weights;
  text_model
;;

let build_vae vae_weights device =
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
