open Torch

(* pub fn build_clip_transformer( *)
(*     clip_weights: &str, *)
(*     device: Device, *)
(* ) -> anyhow::Result<clip::ClipTextTransformer> { *)
(*     let mut vs = nn::VarStore::new(device); *)
(*     let text_model = clip::ClipTextTransformer::new(vs.root()); *)
(*     vs.load(clip_weights)?; *)
(*     Ok(text_model) *)
(* } *)

let build_clip_transformer ~clip_weights ~device =
  let vs = Var_store.create ~device ~name:"" () in
  let _text_model = Diffusers_transformers.Clip.ClipTextTransformer.make vs in
  let named_tensors = Var_store.all_vars vs in
  List.iter (fun (n, _t) -> Printf.printf "Named %s\n" n) named_tensors;
  Serialize.load_multi_ ~named_tensors ~filename:clip_weights
;;
