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

  let make () =
    { num_layers = 1
    ; resnet_eps = 1e-6
    ; resnet_groups = 32
    ; output_scale_factor = 1.
    ; add_downsample = true
    ; downsample_padding = 1
    }
  ;;
end

module DownEncoderBlock2D = struct
  type t =
    { resnets : Resnet.ResnetBlock2D.t list
    ; downsampler : Downsample2D.t option
    ; config : DownEncoderBlock2DConfig.t
    }

  let make vs in_channels out_channels (config : DownEncoderBlock2DConfig.t) =
    let resnets =
      let vs = Var_store.(vs / "resnets") in
      let cfg = Resnet.ResnetBlock2DConfig.default () in
      let cfg =
        { cfg with
          out_channels = Some out_channels
        ; eps = config.resnet_eps
        ; groups = config.resnet_groups
        ; output_scale_factor = config.output_scale_factor
        ; temb_channels = None
        }
      in
      List.init config.num_layers (fun i ->
        let in_channels = if i == 0 then in_channels else out_channels in
        Resnet.ResnetBlock2D.make Var_store.(vs // i) in_channels cfg)
    in
    let downsampler =
      if config.add_downsample
      then (
        let downsample =
          Downsample2D.make
            Var_store.(vs / "downsamplers" // 0)
            out_channels
            true
            out_channels
            config.downsample_padding
        in
        Some downsample)
      else None
    in
    { resnets; downsampler; config }
  ;;

  let forward t xs =
    let xs =
      Base.List.fold t.resnets ~init:xs ~f:(fun acc r ->
        Resnet.ResnetBlock2D.forward r acc None)
    in
    Base.Option.fold t.downsampler ~init:xs ~f:(fun acc downsampler ->
      Downsample2D.forward downsampler acc)
  ;;
end

module UpDecoderBlock2DConfig = struct
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_upsample : bool
    }

  let default () =
    { num_layers = 1
    ; resnet_eps = 1e-6
    ; resnet_groups = 32
    ; output_scale_factor = 1.
    ; add_upsample = true
    }
  ;;
end

module UpDecoderBlock2D = struct
  type t =
    { resnets : Resnet.ResnetBlock2D.t list
    ; upsampler : Upsample2D.t option
    ; config : UpDecoderBlock2DConfig.t
    }

  let make vs in_channels out_channels (config : UpDecoderBlock2DConfig.t) =
    let resnets =
      let vs = Var_store.(vs / "resnets") in
      let cfg = Resnet.ResnetBlock2DConfig.default () in
      let cfg =
        { cfg with
          out_channels = Some out_channels
        ; eps = config.resnet_eps
        ; groups = config.resnet_groups
        ; output_scale_factor = config.output_scale_factor
        ; temb_channels = None
        }
      in
      List.init config.num_layers (fun i ->
        let in_channels = if i == 0 then in_channels else out_channels in
        Resnet.ResnetBlock2D.make Var_store.(vs // i) in_channels cfg)
    in
    let upsampler =
      if config.add_upsample
      then
        Some
          (Upsample2D.make Var_store.(vs / "upsamplers" // 0) out_channels out_channels)
      else None
    in
    { resnets; upsampler; config }
  ;;

  let forward t xs =
    let xs =
      Base.List.fold t.resnets ~init:xs ~f:(fun acc r ->
        Resnet.ResnetBlock2D.forward r acc None)
    in
    Base.Option.fold t.upsampler ~init:xs ~f:(fun acc upsampler ->
      Upsample2D.forward upsampler acc None)
  ;;
end

module UNetMidBlock2DConfig = struct
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int option
    ; attn_num_head_channels : int option
    ; output_scale_factor : float
    }

  let default () =
    { num_layers = 1
    ; resnet_eps = 1e-6
    ; resnet_groups = Some 32
    ; attn_num_head_channels = Some 1
    ; output_scale_factor = 1.
    }
  ;;
end

module UNetMidBlock2D = struct
  type t =
    { resnet : Resnet.ResnetBlock2D.t
    ; attn_resnets : (Attention.AttentionBlock.t * Resnet.ResnetBlock2D.t) list
    ; config : UNetMidBlock2DConfig.t
    }

  let make vs in_channels temb_channels (config : UNetMidBlock2DConfig.t) =
    let vs_resnets = Var_store.(vs / "resnets") in
    let vs_attns = Var_store.(vs / "attentions") in
    let resnet_groups =
      Option.value
        config.resnet_groups
        ~default:(Base.Int.min Base.Int.(in_channels / 4) 32)
    in
    let resnet_cfg = Resnet.ResnetBlock2DConfig.default () in
    let resnet_cfg =
      { resnet_cfg with
        eps = config.resnet_eps
      ; groups = resnet_groups
      ; output_scale_factor = config.output_scale_factor
      ; temb_channels
      }
    in
    let resnet =
      Resnet.ResnetBlock2D.make Var_store.(vs_resnets / "0") in_channels resnet_cfg
    in
    let attn_cfg =
      Attention.AttentionBlockConfig.
        { num_head_channels = config.attn_num_head_channels
        ; num_groups = resnet_groups
        ; rescale_output_factor = config.output_scale_factor
        ; eps = config.resnet_eps
        }
    in
    let attn_resnets =
      List.init config.num_layers (fun i ->
        let attn =
          Attention.AttentionBlock.make Var_store.(vs_attns // i) in_channels attn_cfg
        in
        let resnet =
          Resnet.ResnetBlock2D.make Var_store.(vs_resnets // i) in_channels resnet_cfg
        in
        attn, resnet)
    in
    { resnet; attn_resnets; config }
  ;;

  let forward t xs temb =
    let xs = Resnet.ResnetBlock2D.forward t.resnet xs temb in
    Base.List.fold t.attn_resnets ~init:xs ~f:(fun xs (attn, resnet) ->
      Resnet.ResnetBlock2D.forward resnet (Attention.AttentionBlock.forward attn xs) temb)
  ;;
end
