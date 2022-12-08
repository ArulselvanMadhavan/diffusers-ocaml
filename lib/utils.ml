open Torch

let cpu_or_cuda cpu_components name =
  if Base.List.exists cpu_components ~f:(fun c -> c = "all" || c = name)
  then Device.Cpu
  else Device.cuda_if_available ()
;;

let array_to_tensor tokens device =
  let tokens =
    Bigarray.Array1.of_array Bigarray.Int Bigarray.C_layout (Array.of_list tokens)
  in
  let tokens = Tensor.of_bigarray ~device (Bigarray.genarray_of_array1 tokens) in
  Tensor.view tokens ~size:[ 1; -1 ]
;;

let gen_tokens prompt clip_device =
  let open Diffusers_transformers in
  let tokenizer = Clip.Tokenizer.make "data/bpe_simple_vocab_16e6.txt" in
  let tokens = Clip.Tokenizer.encode tokenizer prompt in
  let tokens = array_to_tensor tokens clip_device in
  let uncond_tokens = Clip.Tokenizer.encode tokenizer "" in
  let uncond_tokens = array_to_tensor uncond_tokens clip_device in
  tokens, uncond_tokens
;;

let build_text_embeddings clip_weights clip_device tokens uncond_tokens =
  let open Diffusers_pipelines in
  let open Diffusers_transformers in
  let text_model =
    Stable_diffusion.build_clip_transformer ~clip_weights ~device:clip_device
  in
  let text_embeddings = Clip.ClipTextTransformer.forward text_model tokens in
  let uncond_embeddings = Clip.ClipTextTransformer.forward text_model uncond_tokens in
  Tensor.cat [ uncond_embeddings; text_embeddings ] ~dim:0
;;

let guidance_scale = 7.5

let update_latents gen_unet_inputs latents unet timestep text_embeddings scheduler =
  let open Diffusers_models in
  let open Diffusers_schedulers in
  let noise_pred =
    Unet_2d.UNet2DConditionModel.forward
      unet
      (gen_unet_inputs latents)
      (Float.of_int timestep)
      text_embeddings
  in
  let noise_pred = Tensor.chunk noise_pred ~chunks:2 ~dim:0 in
  let noise_pred = Array.of_list noise_pred in
  let noise_pred_uncond = noise_pred.(0) in
  let noise_pred_text = noise_pred.(1) in
  let noise_pred =
    Tensor.(
      noise_pred_uncond
      + mul_scalar (noise_pred_text - noise_pred_uncond) (Scalar.f guidance_scale))
  in
  Ddim.DDimScheduler.step scheduler noise_pred timestep latents
;;

let build_image idx num_samples vae_device vae latents final_image =
  let open Diffusers_models in
  Printf.printf "Generating final for sample %d/%d\n" (idx + 1) num_samples;
  let latents = Tensor.to_device ~device:vae_device latents in
  let image =
    Vae.AutoEncoderKL.decode vae (Tensor.div_scalar latents (Scalar.f 0.18215))
  in
  let image = Tensor.(add_scalar (div_scalar image (Scalar.f 2.)) (Scalar.f 0.5)) in
  let image = Tensor.clamp ~min:(Scalar.f 0.) ~max:(Scalar.f 1.) image in
  let image = Tensor.to_device image ~device:Device.Cpu in
  let image = Tensor.(mul_scalar image (Scalar.f 255.)) in
  let image = Tensor.to_kind image ~kind:(T Uint8) in
  let final_image =
    if num_samples > 1
    then (
      match Base.String.rsplit2 ~on:'.' final_image with
      | None -> Printf.sprintf "%s.%s.png" final_image (Int.to_string (idx + 1))
      | Some (filename_no_extension, extension) ->
        Printf.sprintf
          "%s.%s.%s"
          filename_no_extension
          (Int.to_string (idx + 1))
          extension)
    else final_image
  in
  Torch_vision.Image.write_image image ~filename:final_image
;;
