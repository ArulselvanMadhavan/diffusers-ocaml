open Torch

let run_stable_diffusion prompt cpu =
  let cuda_device = Torch.Device.cuda_if_available () in
  let cpu_or_cuda name =
    if Base.List.exists cpu ~f:(fun c -> c == "all" || c == name)
    then Device.Cpu
    else cuda_device
  in
  let clip_device = cpu_or_cuda "clip" in
  let vae_device = cpu_or_cuda "vae" in
  let unet_device = cpu_or_cuda "unet" in
  List.iter
    (fun d -> Printf.printf "%b\n" (Device.is_cuda d))
    [ clip_device; vae_device; unet_device ];
  let _tokenizer =
    Diffusers_transformers.Clip.Tokenizer.make "data/bpe_simple_vocab_16e6.txt"
  in
  let open Lwt.Syntax in
  let+ _ = Lwt_log.info (Printf.sprintf "Running with prompt:%s\n" prompt) in
  ()
;;

let run_stable_diff prompt cpu = Lwt_main.run (run_stable_diffusion prompt cpu)

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
    Term.(const run_stable_diff $ prompt $ cpu), Cmd.info "generate" ~sdocs:"" ~doc ~man
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
