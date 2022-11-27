open Torch

type t = { apply : Tensor.t -> Tensor.t }

let make vs ~num_groups ~num_channels ~eps ~use_bias =
  let weight =
    Var_store.new_var
      vs
      ~trainable:false
      ~shape:[ num_channels ]
      ~init:Ones
      ~name:"weight"
  in
  let bias =
    if use_bias
    then
      Some
        (Var_store.new_var
           vs
           ~trainable:false
           ~shape:[ num_channels ]
           ~init:Zeros
           ~name:"bias")
    else None
  in
  let apply xs =
    Tensor.group_norm ~num_groups ~weight:(Some weight) ~bias ~eps ~cudnn_enabled:false xs
  in
  { apply }
;;

let forward t xs = t.apply xs
