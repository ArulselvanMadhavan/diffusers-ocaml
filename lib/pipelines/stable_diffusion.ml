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
  let _vs = Var_store.create ~device ~name:"" () in
  let _clip = clip_weights in
  (* let text_model = Diffusers_transformers.Clip.Tokenizer.make *)
  ()
;;
