open Torch

module BetaSchedule : sig
  type t
end

module DDIMSchedulerConfig : sig
  type t =
    { beta_start : float
    ; beta_end : float
    ; beta_schedule : BetaSchedule.t
    ; eta : float
    }

  val default : unit -> t
end

module DDimScheduler : sig
  type t =
    { timesteps : int array
    ; alphas_cumprod :
        (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
    ; step_ratio : int
    ; config : DDIMSchedulerConfig.t
    }

  val make : int -> int -> DDIMSchedulerConfig.t -> t
  val step : t -> Tensor.t -> int -> Tensor.t -> Tensor.t
  val add_noise : t -> Tensor.t -> int -> Tensor.t
end
