let () =
  let resnet_config = Diffusers_models.Resnet.ResnetBlock2DConfig.default () in
  let vs = Torch.Var_store.create ~name:"" () in
  let _resnet = Diffusers_models.Resnet.ResnetBlock2D.make vs 20 resnet_config in
  print_endline "Hello, World!"
;;
