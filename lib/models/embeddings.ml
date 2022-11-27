open Torch

module TimestepEmbedding = struct
  type t =
    { linear_1 : Nn.t
    ; linear_2 : Nn.t
    }

  let make vs channel time_embed_dim =
    let linear_1 =
      Layer.linear Var_store.(vs / "linear_1") ~input_dim:channel time_embed_dim
    in
    let linear_2 =
      Layer.linear Var_store.(vs / "linear_2") ~input_dim:time_embed_dim time_embed_dim
    in
    { linear_1; linear_2 }
  ;;

  let forward t xs =
    let xs = Layer.forward t.linear_1 xs in
    let xs = Tensor.silu xs in
    Layer.forward t.linear_2 xs
  ;;
end

module Timesteps = struct
  type t =
    { num_channels : int
    ; flip_sin_to_cos : bool
    ; downscale_freq_shift : float
    ; device : Device.t
    }

  let make num_channels flip_sin_to_cos downscale_freq_shift device =
    { num_channels; flip_sin_to_cos; downscale_freq_shift; device }
  ;;

  let forward t xs =
    let half_dim = t.num_channels / 2 in
    let exponent = Tensor.arange ~end_:(Scalar.i half_dim) ~options:(T Float, t.device) in
    let exponent =
      Tensor.mul_scalar exponent (Scalar.f (Base.Float.neg (Base.Float.log 10000.)))
    in
    let exponent =
      Tensor.div_scalar
        exponent
        (Scalar.f (Float.of_int half_dim -. t.downscale_freq_shift))
    in
    let emb = Tensor.exp exponent in
    let emb = Tensor.(Tensor.unsqueeze xs ~dim:(-1) * Tensor.unsqueeze emb ~dim:0) in
    let emb =
      if t.flip_sin_to_cos
      then Tensor.cat [ Tensor.cos emb; Tensor.sin emb ] ~dim:(-1)
      else Tensor.cat [ Tensor.sin emb; Tensor.cos emb ] ~dim:(-1)
    in
    if Base.Int.(t.num_channels % 2) == 1
    then Tensor.pad emb ~pad:[ 0; 1; 0; 0 ] ~mode:"constant" ~value:None
    else emb
  ;;
end
