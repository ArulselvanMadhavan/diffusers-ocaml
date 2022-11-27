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
    let key = (Tensor.transpose key ~dim0:(-1) ~dim1:(-2)) in
    let key = Tensor.mul_scalar key (Scalar.f t.scale) in
    let xs = Tensor.matmul query key in
    let xs = Tensor.softmax xs ~dim:(-1) ~dtype:(T Float) in
    let xs = Tensor.matmul xs value in
    reshape_batch_dim_to_heads t xs
end
