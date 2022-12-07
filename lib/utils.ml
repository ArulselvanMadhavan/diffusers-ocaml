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
