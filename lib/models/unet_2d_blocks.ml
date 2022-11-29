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

  let default () =
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
    ; downsampler : Downsample2D.t option (* ; config : DownEncoderBlock2DConfig.t *)
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
    { resnets; downsampler }
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
    ; upsampler : Upsample2D.t option (* ; config : UpDecoderBlock2DConfig.t *)
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
    { resnets; upsampler }
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
        (* ; config : UNetMidBlock2DConfig.t *)
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
    { resnet; attn_resnets }
  ;;

  let forward t xs temb =
    let xs = Resnet.ResnetBlock2D.forward t.resnet xs temb in
    Base.List.fold t.attn_resnets ~init:xs ~f:(fun xs (attn, resnet) ->
      Resnet.ResnetBlock2D.forward resnet (Attention.AttentionBlock.forward attn xs) temb)
  ;;
end

module UNetMidBlock2DCrossAttnConfig = struct
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int option
    ; attn_num_head_channels : int
    ; output_scale_factor : float
    ; cross_attn_dim : int
    ; sliced_attention_size : int option
    }

  let default () =
    { num_layers = 1
    ; resnet_eps = 1e-6
    ; resnet_groups = Some 32
    ; attn_num_head_channels = 1
    ; output_scale_factor = 1.
    ; cross_attn_dim = 1280
    ; sliced_attention_size = None
    }
  ;;
end

module UNetMidBlock2DCrossAttn = struct
  type t =
    { resnet : Resnet.ResnetBlock2D.t
    ; attn_resnets : (Attention.SpatialTransformer.t * Resnet.ResnetBlock2D.t) list
        (* ; config : UNetMidBlock2DCrossAttnConfig.t *)
    }

  let make vs in_channels temb_channels (config : UNetMidBlock2DCrossAttnConfig.t) =
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
    let n_heads = config.attn_num_head_channels in
    let attn_cfg =
      Attention.SpatialTransformerConfig.
        { depth = 1
        ; num_groups = resnet_groups
        ; context_dim = Some config.cross_attn_dim
        ; sliced_attention_size = config.sliced_attention_size
        }
    in
    let attn_resnets =
      List.init config.num_layers (fun i ->
        let attn =
          Attention.SpatialTransformer.make
            Var_store.(vs_attns // i)
            in_channels
            n_heads
            Base.Int.(in_channels / n_heads)
            attn_cfg
        in
        let resnet =
          Resnet.ResnetBlock2D.make
            Var_store.(vs_resnets // (i + 1))
            in_channels
            resnet_cfg
        in
        attn, resnet)
    in
    { resnet; attn_resnets }
  ;;

  let forward t xs temb encoder_hidden_states =
    let xs = Resnet.ResnetBlock2D.forward t.resnet xs temb in
    Base.List.fold t.attn_resnets ~init:xs ~f:(fun xs (attn, resnet) ->
      Resnet.ResnetBlock2D.forward
        resnet
        (Attention.SpatialTransformer.forward attn xs encoder_hidden_states)
        temb)
  ;;
end

module DownBlock2DConfig = struct
  type t =
    { num_layers : int
    ; resnet_eps : float
    ; resnet_groups : int
    ; output_scale_factor : float
    ; add_downsample : bool
    ; downsample_padding : int
    }

  let default () =
    { num_layers = 1
    ; resnet_eps = 1e-6
    ; resnet_groups = 32
    ; output_scale_factor = 1.
    ; add_downsample = true
    ; downsample_padding = 1
    }
  ;;
end

module DownBlock2D = struct
  type t =
    { resnets : Resnet.ResnetBlock2D.t list
    ; downsampler : Downsample2D.t option (* ; config : DownBlock2DConfig.t *)
    }

  let make vs in_channels out_channels temb_channels (config : DownBlock2DConfig.t) =
    let vs_resnets = Var_store.(vs / "resnets") in
    let resnet_cfg = Resnet.ResnetBlock2DConfig.default () in
    let resnet_cfg =
      { resnet_cfg with
        out_channels = Some out_channels
      ; eps = config.resnet_eps
      ; output_scale_factor = config.output_scale_factor
      ; temb_channels
      }
    in
    let resnets =
      List.init config.num_layers (fun i ->
        let in_channels = if i == 0 then in_channels else out_channels in
        Resnet.ResnetBlock2D.make Var_store.(vs_resnets // i) in_channels resnet_cfg)
    in
    let downsampler =
      if config.add_downsample
      then (
        let downsampler =
          Downsample2D.make
            Var_store.(vs / "downsamplers" // 0)
            out_channels
            true
            out_channels
            config.downsample_padding
        in
        Some downsampler)
      else None
    in
    { resnets; downsampler }
  ;;

  let forward t xs temb =
    let xs, os1 =
      Base.List.fold t.resnets ~init:(xs, []) ~f:(fun (xs, os) resnet ->
        let xs = Resnet.ResnetBlock2D.forward resnet xs temb in
        xs, xs :: os)
    in
    let xs, os2 =
      Base.Option.fold t.downsampler ~init:(xs, []) ~f:(fun (xs, os) downsampler ->
        let xs = Downsample2D.forward downsampler xs in
        xs, xs :: os)
    in
    xs, List.concat [ List.rev os1; List.rev os2 ]
  ;;
end

module CrossAttnDownBlock2DConfig = struct
  type t =
    { downblock : DownBlock2DConfig.t
    ; attn_num_head_channels : int
    ; cross_attention_dim : int
    ; sliced_attention_size : int option
    }

  let default () =
    let downblock = DownBlock2DConfig.default () in
    { downblock
    ; attn_num_head_channels = 1
    ; cross_attention_dim = 1280
    ; sliced_attention_size = None
    }
  ;;
end

module CrossAttnDownBlock2D = struct
  type t =
    { downblock : DownBlock2D.t
    ; attentions : Attention.SpatialTransformer.t list
        (* ; config : CrossAttnDownBlock2DConfig.t *)
    }

  let make
    vs
    in_channels
    out_channels
    temb_channels
    (config : CrossAttnDownBlock2DConfig.t)
    =
    let downblock =
      DownBlock2D.make vs in_channels out_channels temb_channels config.downblock
    in
    let n_heads = config.attn_num_head_channels in
    let cfg =
      Attention.SpatialTransformerConfig.
        { depth = 1
        ; context_dim = Some config.cross_attention_dim
        ; num_groups = config.downblock.resnet_groups
        ; sliced_attention_size = config.sliced_attention_size
        }
    in
    let vs_attn = Var_store.(vs / "attentions") in
    let attentions =
      List.init config.downblock.num_layers (fun i ->
        Attention.SpatialTransformer.make
          Var_store.(vs_attn // i)
          out_channels
          n_heads
          (out_channels / n_heads)
          cfg)
    in
    { downblock; attentions }
  ;;

  let forward t xs temb encoder_hidden_states =
    let ra = Base.List.zip_exn t.downblock.resnets t.attentions in
    let xs, os1 =
      Base.List.fold ra ~init:(xs, []) ~f:(fun (xs, os) (resnet, attn) ->
        let xs = Resnet.ResnetBlock2D.forward resnet xs temb in
        let xs = Attention.SpatialTransformer.forward attn xs encoder_hidden_states in
        xs, xs :: os)
    in
    let xs, os2 =
      Base.Option.fold
        t.downblock.downsampler
        ~init:(xs, [])
        ~f:(fun (xs, os) downsampler ->
        let xs = Downsample2D.forward downsampler xs in
        xs, xs :: os)
    in
    xs, List.concat [ List.rev os1; List.rev os2 ]
  ;;
end

module UpBlock2DConfig = struct
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

module UpBlock2D = struct
  type t =
    { resnets : Resnet.ResnetBlock2D.t list
    ; upsampler : Upsample2D.t option (* ; config : UpBlock2DConfig.t *)
    }

  let make
    vs
    in_channels
    prev_output_channels
    out_channels
    temb_channels
    (config : UpBlock2DConfig.t)
    =
    let vs_resnets = Var_store.(vs / "resnets") in
    let resnet_cfg = Resnet.ResnetBlock2DConfig.default () in
    let resnet_cfg =
      { resnet_cfg with
        out_channels = Some out_channels
      ; temb_channels
      ; eps = config.resnet_eps
      ; output_scale_factor = config.output_scale_factor
      }
    in
    let resnets =
      List.init config.num_layers (fun i ->
        let res_skip_channels =
          if i == config.num_layers - 1 then in_channels else out_channels
        in
        let resnet_in_channels = if i == 0 then prev_output_channels else out_channels in
        let in_channels = resnet_in_channels + res_skip_channels in
        Resnet.ResnetBlock2D.make Var_store.(vs_resnets // i) in_channels resnet_cfg)
    in
    let upsampler =
      if config.add_upsample
      then
        Some
          (Upsample2D.make Var_store.(vs / "upsamplers" // 0) out_channels out_channels)
      else None
    in
    { resnets; upsampler }
  ;;

  let forward t xs res_xs temb upsample_size =
    let xs =
      Base.List.foldi t.resnets ~init:xs ~f:(fun index xs resnet ->
        let xs = Tensor.cat ~dim:1 [ xs; res_xs.(Array.length res_xs - index - 1) ] in
        Resnet.ResnetBlock2D.forward resnet xs temb)
    in
    Base.Option.fold t.upsampler ~init:xs ~f:(fun xs upsampler ->
      Upsample2D.forward upsampler xs upsample_size)
  ;;
end

module CrossAttnUpBlock2DConfig = struct
  type t =
    { upblock : UpBlock2DConfig.t
    ; attn_num_head_channels : int
    ; cross_attention_dim : int
    ; sliced_attention_size : int option
    }

  let default () =
    let upblock = UpBlock2DConfig.default () in
    { upblock
    ; attn_num_head_channels = 1
    ; cross_attention_dim = 1280
    ; sliced_attention_size = None
    }
  ;;
end

module CrossAttnUpBlock2D = struct
  type t =
    { upblock : UpBlock2D.t
    ; attentions : Attention.SpatialTransformer.t list
        (* ; config : CrossAttnUpBlock2DConfig.t *)
    }

  let make
    vs
    in_channels
    prev_output_channels
    out_channels
    temb_channels
    (config : CrossAttnUpBlock2DConfig.t)
    =
    let upblock =
      UpBlock2D.make
        vs
        in_channels
        prev_output_channels
        out_channels
        temb_channels
        config.upblock
    in
    let n_heads = config.attn_num_head_channels in
    let cfg =
      Attention.SpatialTransformerConfig.
        { depth = 1
        ; context_dim = Some config.cross_attention_dim
        ; num_groups = config.upblock.resnet_groups
        ; sliced_attention_size = config.sliced_attention_size
        }
    in
    let vs_attn = Var_store.(vs / "attentions") in
    let attentions =
      List.init config.upblock.num_layers (fun i ->
        Attention.SpatialTransformer.make
          Var_store.(vs_attn // i)
          out_channels
          n_heads
          (out_channels / n_heads)
          cfg)
    in
    { attentions; upblock }
  ;;

  let forward t xs res_xs temb upsample_size encoder_hidden_states =
    let attentions = Array.of_list t.attentions in
    let xs =
      Base.List.foldi t.upblock.resnets ~init:xs ~f:(fun index xs resnet ->
        let xs = Tensor.cat [ xs; res_xs.(Array.length res_xs - index - 1) ] ~dim:1 in
        let xs = Resnet.ResnetBlock2D.forward resnet xs temb in
        Attention.SpatialTransformer.forward attentions.(index) xs encoder_hidden_states)
    in
    Base.Option.fold t.upblock.upsampler ~init:xs ~f:(fun xs upsampler ->
      Upsample2D.forward upsampler xs upsample_size)
  ;;
end
