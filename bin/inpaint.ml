open Torch

let height = 512
let width = 512
let version = "0.0.1"

let prepare_mask_and_masked_image in_img_path mask_path =
  let image = Torch_vision.Image.load_image in_img_path in
  let image = Base.Or_error.ok_exn image in
  let image = Tensor.div_scalar image (Scalar.f 255.) in
  let image = Tensor.mul_scalar image (Scalar.f 2.) in
  let image = Tensor.add_scalar image (Scalar.f (-1.)) in
  let mask = Torch_vision.Image.load_image mask_path in
  let mask = Base.Or_error.ok_exn mask in
  let mask = Tensor.mean_dim ~dim:(Some [ 0 ]) ~keepdim:true ~dtype:(T Float) mask in
  let mask = Tensor.ge mask (Scalar.f 122.5) in
  let mask = Tensor.to_dtype mask ~dtype:(T Float) ~non_blocking:true ~copy:false in
  let masked_image = Tensor.(add_scalar (neg mask) (Scalar.i 1)) in
  let masked_image = Tensor.(image * masked_image) in
  Tensor.unsqueeze mask ~dim:0, Tensor.unsqueeze masked_image ~dim:0
;;

let run_inpaint
  input_image
  mask_image
  prompt
  cpu
  clip_weights
  unet_weights
  vae_weights
  sliced_attention_size
  n_steps
  seed
  num_samples
  final_image
  =
  let open Diffusers_ocaml in
  let open Diffusers_schedulers in
  let open Diffusers_pipelines in
  let open Diffusers_models in
  Printf.printf "Cuda available:%b\n" (Cuda.is_available ());
  let mask, masked_image = prepare_mask_and_masked_image input_image mask_image in
  Printf.printf
    "Input_image:%s|Mask_image:%s\n"
    (Tensor.shape_str mask)
    (Tensor.shape_str masked_image);
  let clip_device = Utils.cpu_or_cuda cpu "clip" in
  let vae_device = Utils.cpu_or_cuda cpu "vae" in
  let unet_device = Utils.cpu_or_cuda cpu "unet" in
  let scheduler =
    Ddim.DDimScheduler.make n_steps 1000 (Ddim.DDIMSchedulerConfig.default ())
  in
  let tokens, uncond_tokens = Utils.gen_tokens prompt clip_device in
  Tensor.no_grad (fun _ ->
    Printf.printf "Building Clip Transformer";
    let text_embeddings =
      Utils.build_text_embeddings clip_weights clip_device tokens uncond_tokens
    in
    let text_embeddings = Tensor.to_device ~device:unet_device text_embeddings in
    Printf.printf "Building unet";
    let unet =
      Stable_diffusion.build_unet
        ~unet_weights
        ~device:unet_device
        9
        sliced_attention_size
    in
    let vae = Stable_diffusion.build_vae ~vae_weights ~device:vae_device in
    let mask =
      Tensor.upsample_nearest2d
        ~output_size:[ height / 2; width / 2 ]
        ~scales_h:None
        ~scales_w:None
        mask
    in
    let mask = Tensor.cat [ mask; mask ] ~dim:0 in
    let mask = Tensor.to_device mask ~device:vae_device in
    let masked_image = Tensor.to_device masked_image ~device:vae_device in
    let masked_image_dist = Vae.AutoEncoderKL.encode vae masked_image in
    let bsize = 1 in
    for idx = 0 to num_samples - 1 do
      Torch_core.Wrapper.manual_seed (seed + idx);
      let masked_image_latents =
        Vae.DiagonalGaussianDistribution.sample masked_image_dist
      in
      let masked_image_latents =
        Tensor.(mul_scalar masked_image_latents (Scalar.f 0.18215))
      in
      let masked_image_latents =
        Tensor.to_device ~device:unet_device masked_image_latents
      in
      let masked_image_latents =
        Tensor.cat [ masked_image_latents; masked_image_latents ] ~dim:0
      in
      let latents =
        Tensor.randn
          [ bsize; 4; height / 8; width / 8 ]
          ~device:unet_device
          ~kind:(T Float)
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
        let latent_model_input = Tensor.cat [ !latents; !latents ] ~dim:0 in
        let latent_model_input =
          Tensor.cat [ latent_model_input; mask; masked_image_latents ] ~dim:1
        in
        latents
          := Utils.update_latents
               latent_model_input
               unet
               timestep
               text_embeddings
               scheduler
      done;
      Utils.build_image idx num_samples vae_device vae !latents final_image
    done)
;;

let inpaint
  input_image
  mask_image
  prompt
  cpu
  clip_weights
  vae_weights
  unet_weights
  sliced_attention_size
  n_steps
  seed
  num_samples
  final_image
  =
  let prompt =
    Option.value prompt ~default:"A fantasy landscape, trending on artstation."
  in
  let cpu = Option.value cpu ~default:[ "vae" ] in
  let unet_weights = Option.value unet_weights ~default:"data/unet.ot" in
  let clip_weights = Option.value clip_weights ~default:"data/pytorch_model.ot" in
  let vae_weights = Option.value vae_weights ~default:"data/vae.ot" in
  let n_steps = Option.value n_steps ~default:30 in
  let seed = Option.value seed ~default:32 in
  let num_samples = Option.value num_samples ~default:1 in
  let final_image = Option.value final_image ~default:"sd_final.png" in
  run_inpaint
    input_image
    mask_image
    prompt
    cpu
    clip_weights
    unet_weights
    vae_weights
    sliced_attention_size
    n_steps
    seed
    num_samples
    final_image
;;

let () =
  let open Cmdliner in
  let input_image =
    Arg.(
      required & pos 0 (some string) None & info [] ~docv:"FILE" ~doc:"Input image file")
  in
  let mask_image =
    Arg.(
      required & pos 1 (some string) None & info [] ~docv:"FILE" ~doc:"Input image file")
  in
  let prompt =
    Arg.(
      value
      & opt (some string) None
      & info
          [ "prompt" ]
          ~docv:"PROMPT"
          ~doc:"Prompt text to be used for image generation")
  in
  let cpu =
    Arg.(
      value
      & opt (some (list string)) None
      & info
          [ "cpu" ]
          ~docv:"CPU"
          ~doc:"components to run on cpu. supported:all, clip, vae, unet")
  in
  let clip_weights =
    Arg.(
      value
      & opt (some string) None
      & info [ "clip_weights" ] ~docv:"CLIP_WEIGHTS_FILE" ~doc:"clip weights in ot format")
  in
  let vae_weights =
    Arg.(
      value
      & opt (some string) None
      & info [ "vae_weights" ] ~docv:"VAE_WEIGHTS_FILE" ~doc:"vae weights in ot format")
  in
  let unet_weights =
    Arg.(
      value
      & opt (some string) None
      & info [ "unet_weights" ] ~docv:"UNET_WEIGHTS_FILE" ~doc:"vae weights in ot format")
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
  let final_image =
    Arg.(
      value
      & opt (some string) None
      & info [ "final_image" ] ~docv:"FINAL_IMAGE" ~doc:"final image")
  in
  let doc = "Stable_diffusion: Inpaint" in
  let man = [ `S "DESCRIPTION"; `P "Inpaint" ] in
  let cmd =
    ( Term.(
        const inpaint
        $ input_image
        $ mask_image
        $ prompt
        $ cpu
        $ clip_weights
        $ vae_weights
        $ unet_weights
        $ sliced_attention_size
        $ n_steps
        $ seed
        $ num_samples
        $ final_image)
    , Cmd.info "generate" ~sdocs:"" ~doc ~man )
  in
  let default_cmd = Term.(ret (const (`Help (`Pager, None)))) in
  let info =
    let doc = "Stable Diffusion: Inpaint" in
    Cmd.info "Stable diffusion: Inpaint" ~version ~sdocs:"" ~doc
  in
  let cmds = [ cmd ] |> List.map (fun (cmd, info) -> Cmd.v info cmd) in
  let main_cmd = Cmd.group info ~default:default_cmd cmds in
  Cmd.eval main_cmd |> Caml.exit
;;
