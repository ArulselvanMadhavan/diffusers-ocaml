open Torch
open Torch_vision
(* let guidance_scale = 7.5 *)

let version = "0.0.1"

exception InvalidStrength of string

let image_preprocess input_image =
  let image = Image.load_image input_image in
  let image = Base.Or_error.ok_exn image in
  let _, height, width = Tensor.shape3_exn image in
  let height = height - Base.Int.(height % 32) in
  let width = width - Base.Int.(width % 32) in
  let image = Torch_vision.Image.resize image ~height ~width in
  let image = Tensor.(div_scalar image (Scalar.f 255.)) in
  let image = Tensor.(mul_scalar image (Scalar.f 2.)) in
  let image = Tensor.(add_scalar image (Scalar.i (-1))) in
  Tensor.unsqueeze image ~dim:0
;;

let run_img2img input_image prompt cpu clip_weights unet_weights sliced_attention_size =
  let open Diffusers_ocaml in
  Printf.printf "Cuda available:%b\n" (Cuda.is_available ());
  let clip_device = Utils.cpu_or_cuda cpu "clip" in
  let unet_device = Utils.cpu_or_cuda cpu "unet" in
  let tokens, uncond_tokens = Utils.gen_tokens prompt clip_device in
  Tensor.no_grad (fun _ ->
    Printf.printf "Building Clip Transformer";
    let text_embeddings =
      Utils.build_text_embeddings clip_weights clip_device tokens uncond_tokens
    in
    let _text_embeddings = Tensor.to_device ~device:unet_device text_embeddings in
    Printf.printf "Building unet";
    let _unet =
      Diffusers_pipelines.Stable_diffusion.build_unet
        ~unet_weights
        ~device:unet_device
        4
        sliced_attention_size
    in
    let _init_image = image_preprocess input_image in
    ())
;;

let img2img
  input_image
  prompt
  cpu
  clip_weights
  _vae_weights
  unet_weights
  sliced_attention_size
  _n_steps
  _seed
  _num_samples
  _final_image
  strength
  =
  let prompt =
    Option.value prompt ~default:"A fantasy landscape, trending on artstation."
  in
  let cpu = Option.value cpu ~default:[ "clip"; "unet" ] in
  let unet_weights = Option.value unet_weights ~default:"data/unet.ot" in
  let clip_weights = Option.value clip_weights ~default:"data/pytorch_model.ot" in
  let _vae_weights = Option.value ~default:"data/vae.ot" in
  let _n_steps = Option.value ~default:30 in
  let _seed = Option.value ~default:32 in
  let _num_samples = Option.value ~default:1 in
  let _final_image = Option.value ~default:"sd_final.png" in
  let strength = Option.value strength ~default:0.8 in
  if strength < 0. || strength > 1.
  then raise (InvalidStrength "value must be between 0 and 1")
  else run_img2img input_image prompt cpu clip_weights unet_weights sliced_attention_size
;;

(* let prompt = Option.value prompt ~default:"A fantasy landscape, trending on artstation" in *)

let () =
  let open Cmdliner in
  let input_image =
    Arg.(
      required
      & pos 0 (some string) None
      & info [ "input_image" ] ~docv:"FILE" ~doc:"Input image file")
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
      & opt (some (list string)) (Some [ "all" ])
      & info
          []
          ~docv:"CPU"
          ~doc:"components to run on cpu. supported:all, clip, vae, unet")
  in
  let clip_weights =
    Arg.(
      value
      & opt (some string) None
      & info [] ~docv:"CLIP_WEIGHTS_FILE" ~doc:"clip weights in ot format")
  in
  let vae_weights =
    Arg.(
      value
      & opt (some string) None
      & info [] ~docv:"VAE_WEIGHTS_FILE" ~doc:"vae weights in ot format")
  in
  let unet_weights =
    Arg.(
      value
      & opt (some string) None
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
  let final_image =
    Arg.(value & opt (some string) None & info [] ~docv:"FINAL_IMAGE" ~doc:"final image")
  in
  let strength =
    Arg.(
      value
      & opt (some float) None
      & info
          [ "strength" ]
          ~docv:"STRENGTH"
          ~doc:
            "Strength - value between 0 and 1 indicating how much of initial image \
             information should be discarded")
  in
  let doc = "Stable_diffusion: Image2Image" in
  let man = [ `S "DESCRIPTION"; `P "Image2Image" ] in
  let cmd =
    ( Term.(
        const img2img
        $ input_image
        $ prompt
        $ cpu
        $ clip_weights
        $ vae_weights
        $ unet_weights
        $ sliced_attention_size
        $ n_steps
        $ seed
        $ num_samples
        $ final_image
        $ strength)
    , Cmd.info "img2img" ~sdocs:"" ~doc ~man )
  in
  let default_cmd = Term.(ret (const (`Help (`Pager, None)))) in
  let info =
    let doc = "Stable Diffusion: Image to Image" in
    Cmd.info "Stable diffusion" ~version ~sdocs:"" ~doc
  in
  let cmds = [ cmd ] |> List.map (fun (cmd, info) -> Cmd.v info cmd) in
  let main_cmd = Cmd.group info ~default:default_cmd cmds in
  Cmd.eval main_cmd |> Caml.exit
;;
