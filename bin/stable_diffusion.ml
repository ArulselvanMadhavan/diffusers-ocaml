open Torch
open Diffusers_transformers
module DPipelines = Diffusers_pipelines

let log_device d =
  Base.Fn.(d |> Device.is_cuda |> Printf.sprintf "is_cuda:%b\n" |> Lwt_log.debug)
;;

let set_logger () =
  Lwt_log.default
    := Lwt_log.channel
         ~template:"$(date).$(milliseconds) [$(level)] $(message)"
         ~close_mode:`Keep
         ~channel:Lwt_io.stdout
         ();
  Lwt_log.add_rule "*" Lwt_log.Info
;;

let array_to_tensor tokens device =
  let tokens =
    Bigarray.Array1.of_array Bigarray.Int Bigarray.C_layout (Array.of_list tokens)
  in
  let tokens = Tensor.of_bigarray ~device (Bigarray.genarray_of_array1 tokens) in
  Tensor.view tokens ~size:[ 1; -1 ]
;;

let run_stable_diffusion prompt cpu clip_weights vae_weights =
  let open Lwt.Syntax in
  set_logger ();
  let cuda_device = Torch.Device.cuda_if_available () in
  let cpu_or_cuda name =
    if Base.List.exists cpu ~f:(fun c -> c == "all" || c == name)
    then Device.Cpu
    else cuda_device
  in
  let clip_device = cpu_or_cuda "clip" in
  let vae_device = cpu_or_cuda "vae" in
  let unet_device = cpu_or_cuda "unet" in
  let* _ = Lwt.all @@ List.map log_device [ clip_device; vae_device; unet_device ] in
  let tokenizer = Clip.Tokenizer.make "data/bpe_simple_vocab_16e6.txt" in
  let* _ = Lwt_log.info_f "Running with prompt:%s" prompt in
  let tokens = Clip.Tokenizer.encode tokenizer prompt in
  let tokens = array_to_tensor tokens clip_device in
  let uncond_tokens = Clip.Tokenizer.encode tokenizer "" in
  let uncond_tokens = array_to_tensor uncond_tokens clip_device in
  let* _ = Lwt_log.info "Building clip transformer" in
  let text_model =
    DPipelines.Stable_diffusion.build_clip_transformer ~clip_weights ~device:clip_device
  in
  let text_embeddings = Clip.ClipTextTransformer.forward text_model tokens in
  let uncond_embeddings = Clip.ClipTextTransformer.forward text_model uncond_tokens in
  let _text_embeddings = Tensor.cat [ text_embeddings; uncond_embeddings ] ~dim:0 in
  let _vae = DPipelines.Stable_diffusion.build_vae ~vae_weights ~device:vae_device in
  Lwt.return ()
;;

let exec_stable_diff prompt cpu clip_weights vae_weights =
  Lwt_main.run (run_stable_diffusion prompt cpu clip_weights vae_weights)
;;

let () =
  let open Cmdliner in
  let prompt =
    Arg.(
      required
      & pos 0 (some string) None
      & info [] ~docv:"PROMPT" ~doc:"Prompt text to be used for image generation")
  in
  let cpu =
    Arg.(
      required
      & pos 1 (some (list string)) (Some [ "all" ])
      & info
          []
          ~docv:"CPU"
          ~doc:"components to run on cpu. supported:all, clip, vae, unet")
  in
  let clip_weights =
    Arg.(
      required
      & pos 2 (some string) None
      & info [] ~docv:"CLIP_WEIGHTS_FILE" ~doc:"clip weights in ot format")
  in
  let vae_weights =
    Arg.(
      required
      & pos 3 (some string) None
      & info [] ~docv:"VAE_WEIGHTS_FILE" ~doc:"vae weights in ot format")
  in
  let doc = "Stable_diffusion: Generate image from text" in
  let man = [ `S "DESCRIPTION"; `P "Turn text into image" ] in
  let cmd =
    ( Term.(const exec_stable_diff $ prompt $ cpu $ clip_weights $ vae_weights)
    , Cmd.info "generate" ~sdocs:"" ~doc ~man )
  in
  let default_cmd = Term.(ret (const (`Help (`Pager, None)))) in
  let info =
    let doc = "Stable Diffusion: Text to Image generation" in
    Cmd.info "Stable diffusion" ~version:"0.0.1" ~sdocs:"" ~doc
  in
  let cmds = [ cmd ] |> List.map (fun (cmd, info) -> Cmd.v info cmd) in
  let main_cmd = Cmd.group info ~default:default_cmd cmds in
  Cmd.eval main_cmd |> Caml.exit
;;
