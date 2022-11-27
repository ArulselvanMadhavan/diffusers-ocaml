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

module Upsample2D = struct
  type t = { conv : Nn.t }

  let make vs in_channels out_channels =
    let conv =
      Layer.conv2d
        Var_store.(vs / "conv")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:in_channels
        out_channels
    in
    { conv }
  ;;

  let forward t xs size =
    let xs =
      Option.fold
        size
        ~none:
          (let _bsize, _channels, h, w = Tensor.shape4_exn xs in
           Tensor.upsample_nearest2d
             xs
             ~output_size:[ 2 * h; 2 * w ]
             ~scales_h:(Some 2.)
             ~scales_w:(Some 2.))
        ~some:(fun (h, w) ->
          Tensor.upsample_nearest2d xs ~output_size:[ h; w ] ~scales_h:None ~scales_w:None)
    in
    Layer.forward t.conv xs
  ;;
end

module DownEncoderBlock2DConfig = struct
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_downsample : bool
    ; downsample_padding : int
    }
end
