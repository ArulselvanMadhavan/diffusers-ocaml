open Torch

module ResnetBlock2DConfig = struct
  type t =
    { out_channels : int option
    ; temb_channels : int option
    ; groups : int
    ; groups_out : int option
    ; eps : float
    ; use_in_shortcut : bool option
    ; output_scale_factor : float
    }

  let default () =
    { out_channels = None
    ; temb_channels = Some 512
    ; groups = 32
    ; groups_out = None
    ; eps = 1e-6
    ; use_in_shortcut = None
    ; output_scale_factor = 1.
    }
  ;;
end

module ResnetBlock2D = struct
  type t =
    { norm1 : Group_norm.t
    ; norm2 : Group_norm.t
    ; conv1 : Torch.Nn.t
    ; conv2 : Torch.Nn.t
    ; time_emb_proj : Torch.Nn.t option
    ; conv_shortcut : Torch.Nn.t option
    ; config : ResnetBlock2DConfig.t
    }

  let make (vs : Var_store.t) in_channels (config : ResnetBlock2DConfig.t) =
    let out_channels = Option.value config.out_channels ~default:in_channels in
    let norm1 =
      Group_norm.make
        Var_store.(vs / "norm1")
        ~num_groups:config.groups
        ~num_channels:in_channels
        ~eps:config.eps
        ~use_bias:true
    in
    let conv1 =
      Layer.conv2d
        Var_store.(vs / "conv1")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:in_channels
        out_channels
    in
    let groups_out = Option.value config.groups_out ~default:config.groups in
    let norm2 =
      Group_norm.make
        Var_store.(vs / "norm2")
        ~num_groups:groups_out
        ~num_channels:out_channels
        ~eps:config.eps
        ~use_bias:true
    in
    let conv2 =
      Layer.conv2d
        Var_store.(vs / "conv2")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:out_channels
        out_channels
    in
    let use_in_shortcut =
      Option.value config.use_in_shortcut ~default:(in_channels != out_channels)
    in
    let conv_shortcut =
      if use_in_shortcut
      then
        Some
          (Layer.conv2d
             Var_store.(vs / "conv_shortcut")
             ~ksize:(1, 1)
             ~stride:(1, 1)
             ~padding:(1, 1)
             ~input_dim:in_channels
             out_channels)
      else None
    in
    let time_emb_proj =
      Option.map
        (fun tc ->
          Layer.linear Var_store.(vs / "time_emb_proj") ~input_dim:tc out_channels)
        config.temb_channels
    in
    print_float config.output_scale_factor;
    { norm1; conv1; norm2; conv2; conv_shortcut; time_emb_proj; config }
  ;;

  let forward t xs temb =
    let shortcut_xs =
      Option.fold
        ~none:xs
        ~some:(fun conv_shortcut -> Layer.forward conv_shortcut xs)
        t.conv_shortcut
    in
    let xs = Group_norm.forward t.norm1 xs in
    let xs = Tensor.silu xs in
    let xs = Layer.forward t.conv1 xs in
    let xs =
      match temb, t.time_emb_proj with
      | Some temb, Some time_emb_proj ->
        let temb = Tensor.silu temb in
        let temb = Layer.forward time_emb_proj temb in
        let temb = Tensor.unsqueeze temb ~dim:(-1) in
        let temb = Tensor.unsqueeze temb ~dim:(-1) in
        Tensor.add xs temb
      | _ -> xs
    in
    let xs = Tensor.silu (Group_norm.forward t.norm2 xs) in
    let xs = Layer.forward t.conv2 xs in
    Tensor.div_scalar (Tensor.add xs shortcut_xs) (Scalar.f t.config.output_scale_factor)
  ;;

  let print_resnet t =
    let _ = t.norm1 in
    let _ = t.config in
    let _ = t.conv1 in
    let _ = t.norm2 in
    let _ = t.conv2 in
    let _ = t.conv_shortcut in
    let _ = t.time_emb_proj in
    ()
  ;;
end
