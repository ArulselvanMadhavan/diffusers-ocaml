open Torch
open Unet_2d_blocks

module EncoderConfig = struct
  type t =
    { block_out_channels : int array
    ; layers_per_block : int
    ; norm_num_groups : int
    ; double_z : bool
    }

  let default () =
    { block_out_channels = [| 64 |]
    ; layers_per_block = 2
    ; norm_num_groups = 32
    ; double_z = true
    }
  ;;
end

module Encoder = struct
  type t =
    { conv_in : Nn.t
    ; down_blocks : DownEncoderBlock2D.t array
    ; mid_block : UNetMidBlock2D.t
    ; conv_norm_out : Group_norm.t
    ; conv_out : Nn.t (* ; config : EncoderConfig.t *)
    }

  let make vs in_channels out_channels (config : EncoderConfig.t) =
    let conv_in =
      Layer.conv2d
        Var_store.(vs / "conv_in")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:in_channels
        config.block_out_channels.(0)
    in
    let vs_down_blocks = Var_store.(vs / "down_blocks") in
    let down_blocks =
      Array.init (Array.length config.block_out_channels) (fun index ->
        let out_channels = config.block_out_channels.(index) in
        let in_channels =
          if index > 0
          then config.block_out_channels.(index - 1)
          else config.block_out_channels.(0)
        in
        let is_final = index == Array.length config.block_out_channels - 1 in
        let cfg = DownEncoderBlock2DConfig.default () in
        let cfg =
          { cfg with
            num_layers = config.layers_per_block
          ; resnet_eps = 1e-6
          ; resnet_groups = config.norm_num_groups
          ; add_downsample = not is_final
          ; downsample_padding = 0
          }
        in
        DownEncoderBlock2D.make
          Var_store.(vs_down_blocks // index)
          in_channels
          out_channels
          cfg)
    in
    let last_block_out_channels = Base.Array.last config.block_out_channels in
    let mid_cfg = UNetMidBlock2DConfig.default () in
    let mid_cfg =
      { mid_cfg with
        resnet_eps = 1e-6
      ; output_scale_factor = 1.
      ; attn_num_head_channels = None
      ; resnet_groups = Some config.norm_num_groups
      }
    in
    let mid_block =
      UNetMidBlock2D.make
        Var_store.(vs / "mid_block")
        last_block_out_channels
        None
        mid_cfg
    in
    let conv_norm_out =
      Group_norm.make
        Var_store.(vs / "conv_norm_out")
        ~num_groups:config.norm_num_groups
        ~num_channels:last_block_out_channels
        ~eps:1e-6
        ~use_bias:true
    in
    let conv_out_channels = if config.double_z then 2 * out_channels else out_channels in
    let conv_out =
      Layer.conv2d
        Var_store.(vs / "conv_out")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:last_block_out_channels
        conv_out_channels
    in
    { conv_in; down_blocks; mid_block; conv_norm_out; conv_out }
  ;;

  let forward t xs =
    let xs = Layer.forward t.conv_in xs in
    let xs =
      Base.Array.fold t.down_blocks ~init:xs ~f:(fun xs down_block ->
        DownEncoderBlock2D.forward down_block xs)
    in
    let xs = UNetMidBlock2D.forward t.mid_block xs None in
    let xs = Group_norm.forward t.conv_norm_out xs in
    let xs = Tensor.silu xs in
    Layer.forward t.conv_out xs
  ;;
end

module DecoderConfig = struct
  type t =
    { block_out_channels : int list
    ; layers_per_block : int
    ; norm_num_groups : int
    }

  let default () =
    { block_out_channels = [ 64 ]; layers_per_block = 2; norm_num_groups = 32 }
  ;;
end

module Decoder = struct
  type t =
    { conv_in : Nn.t
    ; up_blocks : UpDecoderBlock2D.t list
    ; mid_block : UNetMidBlock2D.t
    ; conv_norm_out : Group_norm.t
    ; conv_out : Nn.t (* ; config : DecoderConfig.t *)
    }

  let make vs in_channels out_channels (config : DecoderConfig.t) =
    let n_block_out_channels = List.length config.block_out_channels in
    let last_block_out_channels = Base.List.last_exn config.block_out_channels in
    let conv_in =
      Layer.conv2d
        Var_store.(vs / "conv_in")
        ~input_dim:in_channels
        last_block_out_channels
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
    in
    let mid_cfg =
      UNetMidBlock2DConfig.
        { (UNetMidBlock2DConfig.default ()) with
          resnet_eps = 1e-6
        ; output_scale_factor = 1.
        ; attn_num_head_channels = None
        ; resnet_groups = Some config.norm_num_groups
        }
    in
    let mid_block =
      UNetMidBlock2D.make
        Var_store.(vs / "mid_block")
        last_block_out_channels
        None
        mid_cfg
    in
    let vs_up_blocks = Var_store.(vs / "up_blocks") in
    let reversed_block_out_channels =
      Array.of_list (List.rev config.block_out_channels)
    in
    let up_blocks =
      List.init n_block_out_channels (fun index ->
        let out_channels = reversed_block_out_channels.(index) in
        let in_channels =
          if index > 0
          then reversed_block_out_channels.(index - 1)
          else reversed_block_out_channels.(0)
        in
        let is_final = index + 1 == n_block_out_channels in
        let cfg = UpDecoderBlock2DConfig.default () in
        let cfg =
          { cfg with
            num_layers = config.layers_per_block + 1
          ; resnet_eps = 1e-6
          ; resnet_groups = config.norm_num_groups
          ; add_upsample = not is_final
          }
        in
        UpDecoderBlock2D.make
          Var_store.(vs_up_blocks // index)
          in_channels
          out_channels
          cfg)
    in
    let conv_norm_out =
      Group_norm.make
        Var_store.(vs / "conv_norm_out")
        ~num_groups:config.norm_num_groups
        ~num_channels:(List.hd config.block_out_channels)
        ~eps:1e-6
        ~use_bias:true
    in
    let conv_out =
      Layer.conv2d
        Var_store.(vs / "conv_out")
        ~input_dim:(List.hd config.block_out_channels)
        out_channels
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
    in
    { conv_in; conv_out; conv_norm_out; up_blocks; mid_block }
  ;;

  let forward t xs =
    let xs = Layer.forward t.conv_in xs in
    let xs = UNetMidBlock2D.forward t.mid_block xs None in
    let xs =
      Base.List.fold t.up_blocks ~init:xs ~f:(fun xs up_block ->
        UpDecoderBlock2D.forward up_block xs)
    in
    let xs = Group_norm.forward t.conv_norm_out xs in
    let xs = Tensor.silu xs in
    Layer.forward t.conv_out xs
  ;;
end

module AutoEncoderKLConfig = struct
  type t =
    { block_out_channels : int list
    ; layers_per_block : int
    ; latent_channels : int
    ; norm_num_groups : int
    }

  let default () =
    { block_out_channels = [ 64 ]
    ; layers_per_block = 1
    ; latent_channels = 4
    ; norm_num_groups = 32
    }
  ;;
end

module DiagonalGaussianDistribution = struct
  type t =
    { mean : Tensor.t
    ; std : Tensor.t
    ; device : Device.t
    }

  let make parameters =
    let parameters = Array.of_list (Tensor.chunk ~chunks:2 ~dim:1 parameters) in
    let mean = parameters.(0) in
    let logvar = parameters.(1) in
    let std = Tensor.mul_scalar logvar (Scalar.f 0.5) in
    let std = Tensor.exp std in
    let device = Tensor.device std in
    { mean; std; device }
  ;;

  let sample t =
    let sample = Tensor.rand_like t.mean in
    let sample = Tensor.to_device ~device:t.device sample in
    Tensor.(t.mean + (t.std * sample))
  ;;
end

module AutoEncoderKL = struct
  type t =
    { encoder : Encoder.t
    ; decoder : Decoder.t
    ; quant_conv : Nn.t
    ; post_quant_conv : Nn.t (* ; config : AutoEncoderKLConfig.t *)
    }

  let make vs in_channels out_channels (config : AutoEncoderKLConfig.t) =
    let latent_channels = config.latent_channels in
    let encoder_cfg =
      EncoderConfig.
        { block_out_channels = Array.of_list config.block_out_channels
        ; layers_per_block = config.layers_per_block
        ; norm_num_groups = config.norm_num_groups
        ; double_z = true
        }
    in
    let encoder =
      Encoder.make Var_store.(vs / "encoder") in_channels latent_channels encoder_cfg
    in
    let decoder_cfg =
      DecoderConfig.
        { block_out_channels = config.block_out_channels
        ; layers_per_block = config.layers_per_block
        ; norm_num_groups = config.norm_num_groups
        }
    in
    let decoder =
      Decoder.make Var_store.(vs / "decoder") latent_channels out_channels decoder_cfg
    in
    let quant_conv =
      Layer.conv2d
        Var_store.(vs / "quant_conv")
        ~ksize:(1, 1)
        ~stride:(1, 1)
        ~padding:(0, 0)
        ~input_dim:(2 * latent_channels)
        (2 * latent_channels)
    in
    let post_quant_conv =
      Layer.conv2d
        Var_store.(vs / "post_quant_conv")
        ~ksize:(1, 1)
        ~stride:(1, 1)
        ~padding:(0, 0)
        ~input_dim:latent_channels
        latent_channels
    in
    { encoder; decoder; quant_conv; post_quant_conv }
  ;;

  let encode t xs =
    let parameters = Encoder.forward t.encoder xs in
    let parameters = Layer.forward t.quant_conv parameters in
    DiagonalGaussianDistribution.make parameters
  ;;

  let decode t xs =
    let xs = Layer.forward t.post_quant_conv xs in
    Decoder.forward t.decoder xs
  ;;
end
