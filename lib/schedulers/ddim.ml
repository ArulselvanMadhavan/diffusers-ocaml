open Torch

module BetaSchedule = struct
  type t =
    | Linear
    | ScaledLinear
end

module DDIMSchedulerConfig = struct
  type t =
    { beta_start : float
    ; beta_end : float
    ; beta_schedule : BetaSchedule.t
    ; eta : float
    }

  let default () =
    let _ = BetaSchedule.Linear in
    { beta_start = 0.00085
    ; beta_end = 0.012
    ; beta_schedule = BetaSchedule.ScaledLinear
    ; eta = 0.
    }
  ;;
end

module DDimScheduler = struct
  type t =
    { timesteps : int array
    ; alphas_cumprod :
        (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
    ; step_ratio : int
    ; config : DDIMSchedulerConfig.t
    }

  let make inference_steps train_timesteps (config : DDIMSchedulerConfig.t) =
    let step_ratio = train_timesteps / inference_steps in
    let timesteps = Array.init (inference_steps + 1) (fun s -> step_ratio * s) in
    Base.Array.rev_inplace timesteps;
    let betas =
      match config.beta_schedule with
      | BetaSchedule.ScaledLinear ->
        Tensor.square
          (Tensor.linspace
             ~start:(Scalar.f config.beta_start)
             ~end_:(Scalar.f config.beta_end)
             ~steps:train_timesteps
             ~options:(T Float, Device.Cpu))
      | BetaSchedule.Linear ->
        Tensor.linspace
          ~start:(Scalar.f config.beta_start)
          ~end_:(Scalar.f config.beta_end)
          ~steps:train_timesteps
          ~options:(T Float, Device.Cpu)
    in
    let alphas = Tensor.(add_scalar (neg betas) (Scalar.f 1.0)) in
    let alphas_cumprod = Tensor.cumprod ~dim:0 ~dtype:(T Double) alphas in
    let alphas_cumprod = Tensor.to_bigarray alphas_cumprod ~kind:Bigarray.Float64 in
    { alphas_cumprod; timesteps; step_ratio; config }
  ;;

  let step t (model_output : Tensor.t) (timestep : int) (sample : Tensor.t) =
    let dims = Bigarray.Genarray.dims t.alphas_cumprod in
    let timestep = if timestep >= dims.(0) then timestep - 1 else timestep in
    let prev_timestep = if timestep > t.step_ratio then timestep - t.step_ratio else 0 in
    let alpha_prod_t = Bigarray.Genarray.get t.alphas_cumprod [| timestep |] in
    let alpha_prod_t_prev = Bigarray.Genarray.get t.alphas_cumprod [| prev_timestep |] in
    let beta_prod_t = 1. -. alpha_prod_t in
    let beta_prod_t_prev = 1. -. alpha_prod_t_prev in
    let pred_original_sample =
      Tensor.(
        div_scalar
          (sample - mul_scalar model_output (Scalar.f (Float.sqrt beta_prod_t)))
          (Scalar.f (Float.sqrt alpha_prod_t)))
    in
    let variance =
      beta_prod_t_prev /. beta_prod_t *. (1. -. (alpha_prod_t /. alpha_prod_t_prev))
    in
    let std_dev_t = t.config.eta *. Float.sqrt variance in
    let pred_sample_direction =
      Tensor.mul_scalar
        model_output
        (Scalar.f (Float.sqrt (1. -. alpha_prod_t_prev -. (std_dev_t *. std_dev_t))))
    in
    let prev_sample =
      Tensor.(
        mul_scalar pred_original_sample (Scalar.f (Float.sqrt alpha_prod_t_prev))
        + pred_sample_direction)
    in
    if t.config.eta > 0.
    then Tensor.(prev_sample + mul_scalar (randn_like prev_sample) (Scalar.f std_dev_t))
    else prev_sample
  ;;
end
