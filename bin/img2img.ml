(* open Torch *)

(* let guidance_scale = 7.5 *)

let version = "0.0.1"

exception InvalidStrength of string

let run_img2img () = ()

let img2img
  prompt
  _cpu
  _clip_weights
  _vae_weights
  _unet_weights
  _sliced_attention_size
  _n_steps
  _seed
  _num_samples
  _final_image
  strength
  =
  let _prompt =
    Option.value prompt ~default:"A fantasy landscape, trending on artstation."
  in
  let strength = Option.value strength ~default:0.8 in
  if strength < 0. || strength > 1.
  then raise (InvalidStrength "value must be between 0 and 1")
  else run_img2img ()
;;

(* let prompt = Option.value prompt ~default:"A fantasy landscape, trending on artstation" in *)

let () =
  let open Cmdliner in
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
