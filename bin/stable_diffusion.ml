open Torch
module DPipelines = Diffusers_pipelines

let height = 512
let width = 512

let run_stable_diffusion
  prompt
  cpu
  clip_weights
  vae_weights
  unet_weights
  sliced_attention_size
  n_steps
  seed
  num_samples
  =
  let open Diffusers_ocaml in
  (* set_logger (); *)
  let n_steps = Option.value n_steps ~default:30 in
  let num_samples = Option.value num_samples ~default:1 in
  let seed = Option.value seed ~default:32 in
  let final_image = "sd_final.png" in
  let clip_device = Utils.cpu_or_cuda cpu "clip" in
  let vae_device = Utils.cpu_or_cuda cpu "vae" in
  let unet_device = Utils.cpu_or_cuda cpu "unet" in
  List.iter
    (fun d -> Printf.printf "%b\n" (Device.is_cuda d))
    [ clip_device; vae_device; unet_device ];
  let scheduler =
    Diffusers_schedulers.Ddim.DDimScheduler.make
      n_steps
      1000
      (Diffusers_schedulers.Ddim.DDIMSchedulerConfig.default ())
  in
  let tokens, uncond_tokens = Utils.gen_tokens prompt clip_device in
  Torch.Tensor.no_grad (fun () ->
    Printf.printf "Building clip transformer\n";
    let text_embeddings =
      Utils.build_text_embeddings clip_weights clip_device tokens uncond_tokens
    in
    let text_embeddings = Tensor.to_device ~device:unet_device text_embeddings in
    Printf.printf "Building unet\n";
    let unet =
      DPipelines.Stable_diffusion.build_unet
        ~unet_weights
        ~device:unet_device
        4
        sliced_attention_size
    in
    let bsize = 1 in
    for idx = 0 to num_samples - 1 do
      Torch_core.Wrapper.manual_seed (seed + idx);
      let latents =
        Tensor.randn
          [ bsize; 4; height / 8; width / 8 ]
          ~kind:(T Float)
          ~device:unet_device
      in
      let latents = ref latents in
      for timestep_index = 0 to Array.length scheduler.timesteps - 1 do
        let timestep = scheduler.timesteps.(timestep_index) in
        Printf.printf
          "Timestep %d/%d|%d|%s\n"
          timestep_index
          n_steps
          timestep
          (Tensor.shape_str !latents);
        Stdio.Out_channel.flush stdout;
        let gen_unet_inputs latents = Tensor.cat [ latents; latents ] ~dim:0 in
        latents
          := Utils.update_latents
               gen_unet_inputs
               !latents
               unet
               timestep
               text_embeddings
               scheduler
      done;
      Printf.printf "Building VAE\n";
      let vae = DPipelines.Stable_diffusion.build_vae ~vae_weights ~device:vae_device in
      Utils.build_image idx num_samples vae_device vae !latents final_image
    done)
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
  let unet_weights =
    Arg.(
      required
      & pos 4 (some string) None
      & info [] ~docv:"UNET_WEIGHTS_FILE" ~doc:"vae weights in ot format")
  in
  let sliced_attention_size =
    Arg.(
      value
      & opt (some int) None
      & info
          [ "sliced_attention_size" ]
          ~docv:"SLICED_ATTENTION_SIZE"
          ~doc:"sliced attention size")
  in
  let n_steps =
    Arg.(
      value
      & opt (some int) None
      & info [ "n_steps" ] ~docv:"N_STEPS" ~doc:"number of timesteps")
  in
  let seed =
    Arg.(
      value
      & opt (some int) None
      & info [ "seed" ] ~docv:"SEED" ~doc:"random seed to be used for generation")
  in
  let num_samples =
    Arg.(
      value
      & opt (some int) None
      & info [ "num_samples" ] ~docv:"NUM_SAMPLES" ~doc:"Number of samples to generate")
  in
  let doc = "Stable_diffusion: Generate image from text" in
  let man = [ `S "DESCRIPTION"; `P "Turn text into image" ] in
  let cmd =
    ( Term.(
        const run_stable_diffusion
        $ prompt
        $ cpu
        $ clip_weights
        $ vae_weights
        $ unet_weights
        $ sliced_attention_size
        $ n_steps
        $ seed
        $ num_samples)
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
