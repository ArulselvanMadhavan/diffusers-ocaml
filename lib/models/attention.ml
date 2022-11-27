open Torch

module GeGlu = struct
  type t = { proj : Nn.t }

  let make vs dim_in dim_out =
    let proj = Nn.linear Var_store.(vs / "proj") ~input_dim:dim_in (dim_out * 2) in
    { proj }
  ;;

  let forward t xs =
    let hidden_states_and_gate = Layer.forward t.proj xs in
    let hidden_states_and_gate =
      Tensor.chunk hidden_states_and_gate ~chunks:2 ~dim:(-1)
    in
    let hsg0 = List.hd hidden_states_and_gate in
    let hsg1 = List.hd (Base.List.drop hidden_states_and_gate 1) in
    let hsg1 = Tensor.gelu hsg1 ~approximate:"none" in
    Tensor.mul hsg0 hsg1
  ;;
end

module Feedforward = struct
  type t =
    { project_in : GeGlu.t
    ; linear : Layer.t
    }

  let make vs dim dim_out mult =
    let inner_dim = dim * mult in
    let dim_out = Option.value dim_out ~default:dim in
    let vs = Var_store.(vs / "net") in
    let project_in = GeGlu.make Var_store.(vs // 0) dim inner_dim in
    let linear = Nn.linear Var_store.(vs // 2) ~input_dim:inner_dim dim_out in
    { project_in; linear }
  ;;

  let forward t xs =
    let xs = GeGlu.forward t.project_in xs in
    Layer.forward t.linear xs
  ;;
end

module CrossAttention = struct
  type t =
    { to_q : Nn.t
    ; to_k : Nn.t
    ; to_v : Nn.t
    ; to_out : Nn.t
    ; heads : int
    ; scale : float
    ; slice_size : int option
    }

  let make vs query_dim context_dim heads dim_head slice_size =
    let inner_dim = dim_head * heads in
    let context_dim = Option.value context_dim ~default:query_dim in
    let scale = Base.Float.(1.0 / Base.Float.sqrt (Float.of_int dim_head)) in
    let to_q =
      Nn.linear Var_store.(vs / "to_q") ~use_bias:false ~input_dim:query_dim inner_dim
    in
    let to_k =
      Nn.linear Var_store.(vs / "to_k") ~use_bias:false ~input_dim:context_dim inner_dim
    in
    let to_v =
      Nn.linear Var_store.(vs / "to_v") ~use_bias:false ~input_dim:context_dim inner_dim
    in
    let to_out =
      Nn.linear Var_store.(vs / "to_out" // 0) ~input_dim:inner_dim query_dim
    in
    { to_q; to_k; to_v; to_out; heads; slice_size; scale }
  ;;

  let reshape_heads_to_batch_dim t xs =
    let batch_size, seq_len, dim = Tensor.shape3_exn xs in
    let xs =
      Tensor.reshape xs ~shape:[ batch_size / t.heads; seq_len; t.heads; dim / t.heads ]
    in
    let xs = Tensor.permute xs ~dims:[ 0; 2; 1; 3 ] in
    Tensor.reshape xs ~shape:[ batch_size / t.heads; seq_len; dim * t.heads ]
  ;;

  let reshape_batch_dim_to_heads t xs =
    let batch_size, seq_len, dim = Tensor.shape3_exn xs in
    let xs = Tensor.reshape xs ~shape:[ batch_size / t.heads; t.heads; seq_len; dim ] in
    let xs = Tensor.permute xs ~dims:[ 0; 2; 1; 3 ] in
    Tensor.reshape xs ~shape:[ batch_size / t.heads; seq_len; dim * t.heads ]
  ;;

  let sliced_attention t query key value sequence_length dim slice_size =
    let batch_size_attention = List.hd (Tensor.size query) in
    let hidden_states =
      Tensor.zeros
        [ batch_size_attention; sequence_length; dim / t.heads ]
        ~kind:(Tensor.kind query)
        ~device:(Tensor.device query)
    in
    let hidden_states = ref hidden_states in
    for i = 0 to (batch_size_attention / slice_size) - 1 do
      let start_idx = i * slice_size in
      let end_idx = (i + 1) * slice_size in
      let query =
        Tensor.slice query ~dim:0 ~start:(Some start_idx) ~end_:(Some end_idx) ~step:1
      in
      let key =
        Tensor.slice key ~dim:0 ~start:(Some start_idx) ~end_:(Some end_idx) ~step:1
      in
      let value =
        Tensor.slice value ~dim:0 ~start:(Some start_idx) ~end_:(Some end_idx) ~step:1
      in
      let key = Tensor.transpose key ~dim0:(-1) ~dim1:(-2) in
      let key = Tensor.mul_scalar key (Scalar.f t.scale) in
      let xs = Tensor.matmul query key in
      let xs = Tensor.softmax xs ~dim:(-1) ~dtype:(T Float) in
      let xs = Tensor.matmul xs value in
      let idx =
        Tensor.arange_start
          ~start:(Scalar.i start_idx)
          ~end_:(Scalar.i end_idx)
          ~options:(T Int64, Tensor.device query)
      in
      hidden_states
        := Tensor.index_put
             !hidden_states
             ~indices:[ Some idx; None; None ]
             ~values:xs
             ~accumulate:false
    done;
    reshape_batch_dim_to_heads t !hidden_states
  ;;

  let attention t query key value =
    let key = Tensor.transpose key ~dim0:(-1) ~dim1:(-2) in
    let key = Tensor.mul_scalar key (Scalar.f t.scale) in
    let xs = Tensor.matmul query key in
    let xs = Tensor.softmax xs ~dim:(-1) ~dtype:(T Float) in
    let xs = Tensor.matmul xs value in
    reshape_batch_dim_to_heads t xs
  ;;

  let forward t xs context =
    let sequence_length = Array.of_list (Tensor.size xs) in
    let sequence_length = sequence_length.(1) in
    let query = Layer.forward t.to_q xs in
    let dim = Base.List.last_exn (Tensor.size query) in
    let context = Option.value context ~default:xs in
    let key = Layer.forward t.to_k context in
    let value = Layer.forward t.to_v context in
    let query = reshape_heads_to_batch_dim t query in
    let key = reshape_heads_to_batch_dim t key in
    let value = reshape_heads_to_batch_dim t value in
    Option.fold
      ~none:(attention t query key value)
      ~some:(fun slice_size ->
        if List.hd (Tensor.size query) / slice_size <= 1
        then attention t query key value
        else
          Layer.forward
            t.to_out
            (sliced_attention t query key value sequence_length dim slice_size))
      t.slice_size
  ;;
end

module BasicTransformerBlock = struct
  type t =
    { attn1 : CrossAttention.t
    ; attn2 : CrossAttention.t
    ; ff : Feedforward.t
    ; norm1 : Nn.t
    ; norm2 : Nn.t
    ; norm3 : Nn.t
    }

  let make vs dim n_heads d_head context_dim sliced_attention_size =
    let attn1 =
      CrossAttention.make
        Var_store.(vs / "attn1")
        dim
        None
        n_heads
        d_head
        sliced_attention_size
    in
    let ff = Feedforward.make Var_store.(vs / "ff") dim None 4 in
    let attn2 =
      CrossAttention.make
        Var_store.(vs / "attn2")
        dim
        context_dim
        n_heads
        d_head
        sliced_attention_size
    in
    let norm1 = Nn.layer_norm Var_store.(vs / "norm1") dim in
    let norm2 = Nn.layer_norm Var_store.(vs / "norm2") dim in
    let norm3 = Nn.layer_norm Var_store.(vs / "norm3") dim in
    { attn1; attn2; ff; norm1; norm2; norm3 }
  ;;

  let forward t xs context =
    let xs1 = Layer.forward t.norm1 xs in
    let xs1 = CrossAttention.forward t.attn1 xs1 None in
    let xs = Tensor.add xs1 xs in
    let xs1 = Layer.forward t.norm2 xs in
    let xs1 = CrossAttention.forward t.attn2 xs1 context in
    let xs = Tensor.add xs1 xs in
    let xs1 = Layer.forward t.norm3 xs in
    let xs1 = Feedforward.forward t.ff xs1 in
    Tensor.add xs1 xs
  ;;
end

module SpatialTransformerConfig = struct
  type t =
    { depth : int
    ; num_groups : int
    ; context_dim : int option
    ; sliced_attention_size : int option
    }

  let make () =
    { depth = 1; num_groups = 32; context_dim = None; sliced_attention_size = None }
  ;;
end

module SpatialTransformer = struct
  type t =
    { norm : Group_norm.t
    ; proj_in : Nn.t
    ; transformer_blocks : BasicTransformerBlock.t list
    ; proj_out : Nn.t
    ; config : SpatialTransformerConfig.t
    }

  let make vs in_channels n_heads d_head (config : SpatialTransformerConfig.t) =
    let inner_dim = n_heads * d_head in
    let num_groups = config.num_groups in
    let norm =
      Group_norm.make
        Var_store.(vs / "norm")
        ~num_groups
        ~num_channels:in_channels
        ~eps:1e-6
        ~use_bias:true
    in
    let proj_in =
      Nn.conv2d
        Var_store.(vs / "proj_in")
        ~ksize:(1, 1)
        ~stride:(1, 1)
        ~padding:(0, 0)
        ~input_dim:in_channels
        inner_dim
    in
    let vs_tb = Var_store.(vs / "transformer_blocks") in
    let transformer_blocks =
      List.init config.depth (fun index ->
        BasicTransformerBlock.make
          Var_store.(vs_tb // index)
          inner_dim
          n_heads
          d_head
          config.context_dim
          config.sliced_attention_size)
    in
    let proj_out =
      Nn.conv2d
        Var_store.(vs / "proj_out")
        ~ksize:(1, 1)
        ~stride:(1, 1)
        ~padding:(0, 0)
        ~input_dim:inner_dim
        in_channels
    in
    { norm; proj_in; transformer_blocks; proj_out; config }
  ;;

  let forward t xs context =
    let (batch, _channel, height, weight) = Tensor.shape4_exn xs in
    let residual = xs in
    let xs = Group_norm.forward t.norm xs in
    let xs = Layer.forward t.proj_in xs in
    let inner_dim = Array.of_list (Tensor.shape xs) in
    let inner_dim = inner_dim.(1) in
    let xs = Tensor.permute xs ~dims:[0;2;3;1] in
    let xs = Tensor.view xs ~size:[batch; height*weight;inner_dim] in
    let xs = List.fold_left (fun acc tb -> BasicTransformerBlock.forward tb acc context) xs t.transformer_blocks in
    let xs = Tensor.view xs ~size:[batch;height;weight;inner_dim] in
    let xs = Tensor.permute xs ~dims:[0;3;1;2] in
    Tensor.add (Layer.forward t.proj_out xs) residual
end
