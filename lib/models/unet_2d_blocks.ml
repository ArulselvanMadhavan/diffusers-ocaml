open Torch

module Downsample2D = struct
  type t =
    { conv : Nn.t option
    ; padding : int
    }

  let make vs in_channels use_conv out_channels padding =
    let conv =
      if use_conv
      then
        Some
          (Layer.conv2d
             Var_store.(vs / "conv")
             ~ksize:(3, 3)
             ~stride:(2, 2)
             ~padding:(padding, padding)
             ~input_dim:in_channels
             out_channels)
      else None
    in
    { conv; padding }
  ;;

  let forward t xs =
    Option.fold
      t.conv
      ~none:
        (Tensor.avg_pool2d
           ~ksize:(2, 2)
           ~stride:(2, 2)
           ~padding:(0, 0)
           ~ceil_mode:false
           ~count_include_pad:true
           xs)
      ~some:(fun conv ->
        if t.padding == 0
        then (
          let result =
            Tensor.pad xs ~pad:[ 0; 1; 0; 1 ] ~mode:"constant" ~value:(Some 0.)
          in
          Layer.forward conv result)
        else Layer.forward conv xs)
  ;;
end
