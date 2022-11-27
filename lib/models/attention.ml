open Torch

module GeGlu = struct
  type t = { proj : Nn.t }

  let make vs dim_in dim_out =
    let proj = Nn.linear Var_store.(vs / "proj") ~input_dim:dim_in (dim_out * 2) in
    { proj }
  ;;

  let forward t xs =
    let hidden_states_and_gate = Layer.forward t.proj xs in
    let hidden_states_and_gate =
      Tensor.chunk hidden_states_and_gate ~chunks:2 ~dim:(-1)
    in
    let hsg0 = List.hd hidden_states_and_gate in
    let hsg1 = List.hd (Base.List.drop hidden_states_and_gate 1) in
    let hsg1 = Tensor.gelu hsg1 ~approximate:"none" in
    Tensor.mul hsg0 hsg1
  ;;
end

module Feedforward = struct
  type t =
    { project_in : GeGlu.t
    ; linear : Layer.t
    }

  let make vs dim dim_out mult =
    let inner_dim = dim * mult in
    let dim_out = Option.value dim_out ~default:dim in
    let vs = Var_store.(vs / "net") in
    let project_in = GeGlu.make Var_store.(vs / "0") dim inner_dim in
    let linear = Nn.linear Var_store.(vs / "2") ~input_dim:inner_dim dim_out in
    { project_in; linear }
  ;;

  let forward t xs =
    let xs = GeGlu.forward t.project_in xs in
    Layer.forward t.linear xs
  ;;
end
