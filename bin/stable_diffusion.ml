open Torch
open Diffusers_transformers

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

let run_stable_diffusion prompt cpu =
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
  let tokens =
    Bigarray.Array1.of_array Bigarray.Int Bigarray.C_layout (Array.of_list tokens)
  in
  let _tokens =
    Tensor.of_bigarray ~device:clip_device (Bigarray.genarray_of_array1 tokens)
  in
  let uncond_tokens = Clip.Tokenizer.encode tokenizer "" in
  Printf.printf "uncond_tokens:%d\n" (List.length uncond_tokens);
  List.iter (Printf.printf "token:%d\n") (Base.List.take uncond_tokens 10);
  (* let _tokens = Tensor.of_bigarray ~device:clip_device tokens in *)
  (* () *)
  Lwt.return ()
;;

let exec_stable_diff prompt cpu = Lwt_main.run (run_stable_diffusion prompt cpu)

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
  let doc = "Stable_diffusion: Generate image from text" in
  let man = [ `S "DESCRIPTION"; `P "Turn text into image" ] in
  let cmd =
    Term.(const exec_stable_diff $ prompt $ cpu), Cmd.info "generate" ~sdocs:"" ~doc ~man
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
