open Torch
open Diffusers_transformers
module DPipelines = Diffusers_pipelines

let height = 512
let width = 512
let guidance_scale = 7.5

(* let log_device d = *)
(*   Base.Fn.(d |> Device.is_cuda |> Printf.sprintf "is_cuda:%b\n" |> Lwt_log.debug) *)

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

let gen_tokens prompt clip_device =
  let tokenizer = Clip.Tokenizer.make "data/bpe_simple_vocab_16e6.txt" in
  let tokens = Clip.Tokenizer.encode tokenizer prompt in
  let tokens = array_to_tensor tokens clip_device in
  let uncond_tokens = Clip.Tokenizer.encode tokenizer "" in
  let uncond_tokens = array_to_tensor uncond_tokens clip_device in
  tokens, uncond_tokens
;;

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
  let open Diffusers_models in
  set_logger ();
  let n_steps = Option.value n_steps ~default:30 in
  let num_samples = Option.value num_samples ~default:1 in
  let seed = Option.value seed ~default:32 in
  let final_image = "sd_final.png" in
  let cuda_device = Torch.Device.cuda_if_available () in
  let cpu_or_cuda name =
    if Base.List.exists cpu ~f:(fun c -> c == "all" || c == name)
    then Device.Cpu
    else cuda_device
  in
  let clip_device = cpu_or_cuda "clip" in
  let vae_device = cpu_or_cuda "vae" in
  let unet_device = cpu_or_cuda "unet" in
  (* let* _ = Lwt.all @@ List.map log_device [ clip_device; vae_device; unet_device ] in *)
  let scheduler =
    Diffusers_schedulers.Ddim.DDimScheduler.make
      n_steps
      1000
      (Diffusers_schedulers.Ddim.DDIMSchedulerConfig.default ())
  in
  let tokens, uncond_tokens = gen_tokens prompt clip_device in
  Torch.Tensor.no_grad (fun () ->
    Printf.printf "Building clip transformer\n";
    let text_model =
      DPipelines.Stable_diffusion.build_clip_transformer ~clip_weights ~device:clip_device
    in
    let text_embeddings = Clip.ClipTextTransformer.forward text_model tokens in
    let uncond_embeddings = Clip.ClipTextTransformer.forward text_model uncond_tokens in
    let text_embeddings = Tensor.cat [ uncond_embeddings; text_embeddings ] ~dim:0 in
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
        Printf.printf "Timestep %d/%d|%d\n" timestep_index n_steps timestep;
        Stdio.Out_channel.flush stdout;
        let latent_model_input = Tensor.cat [ !latents; !latents ] ~dim:0 in
        let noise_pred =
          Unet_2d.UNet2DConditionModel.forward
            unet
            latent_model_input
            (Float.of_int timestep)
            text_embeddings
        in
        let noise_pred = Array.of_list (Tensor.chunk noise_pred ~chunks:2 ~dim:0) in
        let noise_pred =
          Tensor.(
            noise_pred.(0)
            + mul_scalar (noise_pred.(1) - noise_pred.(0)) (Scalar.f guidance_scale))
        in
        latents
          := Diffusers_schedulers.Ddim.DDimScheduler.step
               scheduler
               noise_pred
               timestep
               !latents;
        Caml.Gc.full_major ()
      done;
      Printf.printf "Generating final for sample %d/%d\n" (idx + 1) num_samples;
      let latents = Tensor.to_device ~device:vae_device !latents in
      Printf.printf "Building VAE\n";
      let vae = DPipelines.Stable_diffusion.build_vae ~vae_weights ~device:vae_device in
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
