open Torch

let build_clip_transformer ~clip_weights ~device =
  let vs = Var_store.create ~device ~name:"" () in
  let text_model = Diffusers_transformers.Clip.ClipTextTransformer.make vs in
  let named_tensors = Var_store.all_vars vs in
  Serialize.load_multi_ ~named_tensors ~filename:clip_weights;
  text_model
;;
