let () =
  let resnet_config = Diffusers_models.Resnet.make_config () in
  let vs = Torch.Var_store.create ~name:"" () in
  let _resnet = Diffusers_models.Resnet.make vs 20 resnet_config in
  print_endline "Hello, World!"
;;
