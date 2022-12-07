(* open Torch *)

let version = "0.0.1"

let inpaint
  _input_image
  _prompt
  _cpu
  _clip_weights
  _unet_weights
  _vae_weights
  _sliced_attention_size
  _n_steps
  _strength
  _seed
  _num_samples
  _final_image
  =
  ()
;;

let () =
  let open Cmdliner in
  let input_image =
    Arg.(
      required & pos 0 (some string) None & info [] ~docv:"FILE" ~doc:"Input image file")
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
  let doc = "Stable_diffusion: Inpaint" in
  let man = [ `S "DESCRIPTION"; `P "Inpaint" ] in
  let cmd =
    ( Term.(
        const inpaint
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
    , Cmd.info "inpaint" ~sdocs:"" ~doc ~man )
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
