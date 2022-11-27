open Torch

type t = { apply : Tensor.t -> Tensor.t }

let make vs ~num_groups ~num_channels ~eps =
  let weight =
    Var_store.new_var
      vs
      ~trainable:false
      ~shape:[ num_channels ]
      ~init:Ones
      ~name:"weight"
  in
  let bias =
    Var_store.new_var vs ~trainable:false ~shape:[ num_channels ] ~init:Zeros ~name:"bias"
  in
  let apply xs =
    Tensor.group_norm
      ~num_groups
      ~weight:(Some weight)
      ~bias:(Some bias)
      ~eps
      ~cudnn_enabled:false
      xs
  in
  { apply }
;;

let forward t xs = t.apply xs
