open Unet_2d_blocks
open Torch

module BlockConfig = struct
  type t =
    { out_channels : int
    ; use_cross_attn : bool
    }
end

module UNet2DConditionModelConfig = struct
  type t =
    { center_input_sample : bool
    ; flip_sin_to_cos : bool
    ; freq_shift : float
    ; blocks : BlockConfig.t array
    ; layers_per_block : int
    ; downsample_padding : int
    ; mid_block_scale_factor : float
    ; norm_num_groups : int
    ; norm_eps : float
    ; cross_attention_dim : int
    ; attention_head_dim : int
    ; sliced_attention_size : int option
    }

  let default () =
    { center_input_sample = false
    ; flip_sin_to_cos = true
    ; freq_shift = 0.
    ; blocks =
        [| BlockConfig.{ out_channels = 320; use_cross_attn = true }
         ; BlockConfig.{ out_channels = 640; use_cross_attn = true }
         ; BlockConfig.{ out_channels = 1280; use_cross_attn = true }
         ; BlockConfig.{ out_channels = 1280; use_cross_attn = false }
        |]
    ; layers_per_block = 2
    ; downsample_padding = 1
    ; mid_block_scale_factor = 1.
    ; norm_num_groups = 32
    ; norm_eps = 1e-5
    ; cross_attention_dim = 1280
    ; attention_head_dim = 8
    ; sliced_attention_size = None
    }
  ;;
end

module UNetDownBlock = struct
  type t =
    | Basic of DownBlock2D.t
    | CrossAttn of CrossAttnDownBlock2D.t
end

module UNetUpBlock = struct
  type t =
    | Basic of UpBlock2D.t
    | CrossAttn of CrossAttnUpBlock2D.t
end

module UNet2DConditionModel = struct
  type t =
    { conv_in : Nn.t
    ; time_proj : Embeddings.Timesteps.t
    ; time_embedding : Embeddings.TimestepEmbedding.t
    ; down_blocks : UNetDownBlock.t list
    ; mid_block : UNetMidBlock2DCrossAttn.t
    ; up_blocks : UNetUpBlock.t list
    ; conv_norm_out : Group_norm.t
    ; conv_out : Nn.t
    ; config : UNet2DConditionModelConfig.t
    }

  let make vs in_channels out_channels (config : UNet2DConditionModelConfig.t) =
    let n_blocks = Array.length config.blocks in
    let b_channels = config.blocks.(0).out_channels in
    let bl_channels = (Base.Array.last config.blocks).out_channels in
    let time_embed_dim = b_channels * 4 in
    let conv_in =
      Layer.conv2d
        Var_store.(vs / "conv_in")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:in_channels
        b_channels
    in
    let time_proj =
      Embeddings.Timesteps.make
        b_channels
        config.flip_sin_to_cos
        config.freq_shift
        (Var_store.device vs)
    in
    let time_embedding =
      Embeddings.TimestepEmbedding.make
        Var_store.(vs / "time_embedding")
        b_channels
        time_embed_dim
    in
    let vs_db = Var_store.(vs / "down_blocks") in
    let down_blocks =
      List.init n_blocks (fun i ->
        let BlockConfig.{ out_channels; use_cross_attn } = config.blocks.(i) in
        let in_channels =
          if i > 0 then config.blocks.(i - 1).out_channels else b_channels
        in
        let db_cfg = DownBlock2DConfig.default () in
        let db_cfg =
          { db_cfg with
            num_layers = config.layers_per_block
          ; resnet_eps = config.norm_eps
          ; resnet_groups = config.norm_num_groups
          ; add_downsample = i < n_blocks - 1
          ; downsample_padding = config.downsample_padding
          }
        in
        if use_cross_attn
        then (
          let config =
            CrossAttnDownBlock2DConfig.
              { downblock = db_cfg
              ; attn_num_head_channels = config.attention_head_dim
              ; cross_attention_dim = config.cross_attention_dim
              ; sliced_attention_size = config.sliced_attention_size
              }
          in
          let block =
            CrossAttnDownBlock2D.make
              Var_store.(vs_db // i)
              in_channels
              out_channels
              (Some time_embed_dim)
              config
          in
          UNetDownBlock.(CrossAttn block))
        else (
          let block =
            DownBlock2D.make
              Var_store.(vs_db // i)
              in_channels
              out_channels
              (Some time_embed_dim)
              db_cfg
          in
          UNetDownBlock.(Basic block)))
    in
    let mid_cfg = UNetMidBlock2DCrossAttnConfig.default () in
    let mid_cfg =
      { mid_cfg with
        resnet_eps = config.norm_eps
      ; output_scale_factor = config.mid_block_scale_factor
      ; cross_attn_dim = config.cross_attention_dim
      ; attn_num_head_channels = config.attention_head_dim
      ; resnet_groups = Some config.norm_num_groups
      }
    in
    let mid_block =
      UNetMidBlock2DCrossAttn.make
        Var_store.(vs / "mid_block")
        bl_channels
        (Some time_embed_dim)
        mid_cfg
    in
    let vs_ub = Var_store.(vs / "up_blocks") in
    let up_blocks =
      List.init n_blocks (fun i ->
        let BlockConfig.{ out_channels; use_cross_attn } =
          config.blocks.(n_blocks - 1 - i)
        in
        let prev_out_channels =
          if i > 0 then config.blocks.(n_blocks - i).out_channels else bl_channels
        in
        let in_channels =
          let index = if i == n_blocks - 1 then 0 else n_blocks - i - 2 in
          config.blocks.(index).out_channels
        in
        let ub_cfg = UpBlock2DConfig.default () in
        let ub_cfg =
          { ub_cfg with
            num_layers = config.layers_per_block + 1
          ; resnet_eps = config.norm_eps
          ; resnet_groups = config.norm_num_groups
          ; add_upsample = i < n_blocks - 1
          }
        in
        if use_cross_attn
        then (
          let config =
            CrossAttnUpBlock2DConfig.
              { upblock = ub_cfg
              ; attn_num_head_channels = config.attention_head_dim
              ; cross_attention_dim = config.cross_attention_dim
              ; sliced_attention_size = config.sliced_attention_size
              }
          in
          let block =
            CrossAttnUpBlock2D.make
              Var_store.(vs_ub // i)
              in_channels
              prev_out_channels
              out_channels
              (Some time_embed_dim)
              config
          in
          UNetUpBlock.CrossAttn block)
        else (
          let block =
            UpBlock2D.make
              Var_store.(vs_ub // i)
              in_channels
              prev_out_channels
              out_channels
              (Some time_embed_dim)
              ub_cfg
          in
          UNetUpBlock.Basic block))
    in
    let conv_norm_out =
      Group_norm.make
        Var_store.(vs / "conv_norm_out")
        ~num_groups:config.norm_num_groups
        ~num_channels:b_channels
        ~eps:config.norm_eps
        ~use_bias:true
    in
    let conv_out =
      Layer.conv2d
        Var_store.(vs / "conv_out")
        ~input_dim:b_channels
        out_channels
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
    in
    { conv_in
    ; time_proj
    ; time_embedding
    ; down_blocks
    ; mid_block
    ; up_blocks
    ; conv_norm_out
    ; conv_out
    ; config
    }
  ;;

  let forward t xs timestep encoder_hidden_states =
    let bsize, _channels, height, width = Tensor.shape4_exn xs in
    let device = Tensor.device xs in
    let n_blocks = Array.length t.config.blocks in
    let num_upsamplers = n_blocks - 1 in
    let default_overall_up_factor = Base.Int.pow 2 num_upsamplers in
    let forward_upsample_size =
      Base.Int.(
        height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0)
    in
    (* center input if necessary *)
    let xs =
      if t.config.center_input_sample
      then Tensor.(sub_scalar (mul_scalar xs (Scalar.f 2.0)) (Scalar.f 1.0))
      else xs
    in
    (* 1. time *)
    let emb = Tensor.ones [ bsize ] ~kind:(T Float) ~device in
    let emb = Tensor.(mul_scalar emb (Scalar.f timestep)) in
    let emb = Embeddings.Timesteps.forward t.time_proj emb in
    let emb = Embeddings.TimestepEmbedding.forward t.time_embedding emb in
    let xs = Layer.forward t.conv_in xs in
    let xs, down_block_res_xs =
      Base.List.fold t.down_blocks ~init:(xs, [ xs ]) ~f:(fun (xs, res_xs) b ->
        let xs, r_xs =
          match b with
          | UNetDownBlock.Basic b ->
            DownBlock2D.forward b xs (Some emb)
          | UNetDownBlock.CrossAttn b ->
            CrossAttnDownBlock2D.forward b xs (Some emb) (Some encoder_hidden_states)
        in
        xs, Base.List.append res_xs r_xs)
    in
    (* mid *)
    let xs =
      UNetMidBlock2DCrossAttn.forward
        t.mid_block
        xs
        (Some emb)
        (Some encoder_hidden_states)
    in
    (* up *)
    let xs, _upsample_size, _down_block_res_xs =
      Base.List.foldi
        t.up_blocks
        ~init:(xs, None, down_block_res_xs)
        ~f:(fun i (xs, upsample_size, down_block_res_xs) b ->
        let n_resnets =
          match b with
          | UNetUpBlock.Basic b -> List.length b.resnets
          | UNetUpBlock.CrossAttn b -> List.length b.upblock.resnets
        in
        (* split off *)
        let split = List.length down_block_res_xs - n_resnets in
        let res_xs = Base.List.drop down_block_res_xs split in
        let down_block_res_xs = Base.List.take down_block_res_xs split in
        let upsample_size =
          if i < n_blocks - 1 && forward_upsample_size
          then (
            let _, _, h, w = Tensor.shape4_exn (Base.List.last_exn down_block_res_xs) in
            Some (h, w))
          else upsample_size
        in
        let xs =
          match b with
          | UNetUpBlock.Basic b ->
            UpBlock2D.forward b xs (Array.of_list res_xs) (Some emb) upsample_size
          | UNetUpBlock.CrossAttn b ->
            CrossAttnUpBlock2D.forward
              b
              xs
              (Array.of_list res_xs)
              (Some emb)
              upsample_size
              (Some encoder_hidden_states)
        in
        xs, upsample_size, down_block_res_xs)
    in
    let xs = Group_norm.forward t.conv_norm_out xs in
    let xs = Tensor.silu xs in
    Layer.forward t.conv_out xs
  ;;
end
