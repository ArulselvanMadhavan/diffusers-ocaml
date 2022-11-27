open Torch

type resnet_block_2dconfig =
  { out_channels : int option
  ; temb_channels : int option
  ; groups : int
  ; groups_out : int option
  ; eps : float
  ; use_in_shortcut : bool option
  ; output_scale_factor : float
  }

let make_config () =
  { out_channels = None
  ; temb_channels = Some 512
  ; groups = 32
  ; groups_out = None
  ; eps = 1e-6
  ; use_in_shortcut = None
  ; output_scale_factor = 1.
  }
;;

type t =
  { norm1 : Group_norm.t
  ; norm2 : Group_norm.t
  ; conv1 : Torch.Nn.t
  ; conv2 : Torch.Nn.t
  ; time_emb_proj : Torch.Nn.t option
  ; conv_shortcut : Torch.Nn.t option
  ; config : resnet_block_2dconfig
  }

let make (vs : Var_store.t) in_channels config =
  let out_channels = Option.value config.out_channels ~default:in_channels in
  let norm1 =
    Group_norm.make
      Var_store.(vs / "norm1")
      ~num_groups:config.groups
      ~num_channels:in_channels
      ~eps:config.eps
  in
  let conv1 =
    Nn.conv2d
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
  in
  let conv2 =
    Nn.conv2d
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
        (Nn.conv2d
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
      (fun tc -> Nn.linear Var_store.(vs / "time_emb_proj") ~input_dim:tc out_channels)
      config.temb_channels
  in
  print_float config.output_scale_factor;
  { norm1; conv1; norm2; conv2; conv_shortcut; time_emb_proj; config }
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
