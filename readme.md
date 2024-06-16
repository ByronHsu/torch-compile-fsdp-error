## Llama3 8B 8 A100s training with FSDP

1. Installation

```bash
$ ./script/100-setup.sh
```

2. Start Training

```bash
$ ./script/200-run.sh
```

3. Error Message

```bash
rank4]: Traceback (most recent call last):
[rank4]:   File "/home/jobuser/resources/liger-recipe/llama3-8b-8-gpu-oss/training.py", line 87, in <module>
[rank4]:     main()
[rank4]:   File "/home/jobuser/resources/liger-recipe/llama3-8b-8-gpu-oss/training.py", line 83, in main
[rank4]:     trainer.train()
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 360, in train
[rank4]:     output = super().train(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/trainer.py", line 1885, in train
[rank4]:     return inner_training_loop(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/trainer.py", line 2216, in _inner_training_loop
[rank4]:     tr_loss_step = self.training_step(model, inputs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/trainer.py", line 3238, in training_step
[rank4]:     loss = self.compute_loss(model, inputs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/trainer.py", line 3264, in compute_loss
[rank4]:     outputs = model(**inputs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/accelerate/utils/operations.py", line 822, in forward
[rank4]:     return model_forward(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/accelerate/utils/operations.py", line 810, in __call__
[rank4]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 43, in decorate_autocast
[rank4]:     return func(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 433, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_dynamo/external_utils.py", line 38, in inner
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 863, in forward
[rank4]:     output = self._fsdp_wrapped_module(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/accelerate/utils/operations.py", line 822, in forward
[rank4]:     return model_forward(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/accelerate/utils/operations.py", line 810, in __call__
[rank4]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 43, in decorate_autocast
[rank4]:     return func(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 1164, in forward
[rank4]:     outputs = self.model(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 940, in forward
[rank4]:     causal_mask = self._update_causal_mask(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 968, in torch_dynamo_resume_in_forward_at_940
[rank4]:     layer_outputs = decoder_layer(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 863, in forward
[rank4]:     output = self._fsdp_wrapped_module(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1566, in _wrapped_call_impl
[rank4]:     return self._call_impl(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1575, in _call_impl
[rank4]:     return forward_call(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py", line 143, in forward
[rank4]:     def forward(self, *args, **kwargs):
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 600, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 996, in forward
[rank4]:     return compiled_fn(full_args)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 204, in runtime_wrapper
[rank4]:     all_outs = call_func_at_runtime_with_args(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 120, in call_func_at_runtime_with_args
[rank4]:     out = normalize_as_list(f(args))
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 94, in g
[rank4]:     return f(*args)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/autograd/function.py", line 574, in apply
[rank4]:     return super().apply(*args, **kwargs)  # type: ignore[misc]
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 1452, in forward
[rank4]:     fw_outs = call_func_at_runtime_with_args(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 120, in call_func_at_runtime_with_args
[rank4]:     out = normalize_as_list(f(args))
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 451, in wrapper
[rank4]:     return compiled_fn(runtime_args)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 1158, in __call__
[rank4]:     return self.current_callable(inputs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 951, in run
[rank4]:     return model(new_inputs)
[rank4]:   File "/tmp/torchinductor_jobuser/3i/c3ilt6srmv7vhz4jupk7oeb4wvwxv75r5qywijr22r73z5r7adsx.py", line 579, in call
[rank4]:     buf12 = torch.ops.higher_order.run_and_save_rng_state(torch.ops.aten._scaled_dot_product_efficient_attention.default, buf9, reinterpret_tensor(buf10, (8, 32, 2048, 128), (8388608, 262144, 128, 1), 0), reinterpret_tensor(buf11, (8, 32, 2048, 128), (8388608, 262144, 128, 1), 0), reinterpret_tensor(primals_2, (8, 32, 2048, 2048), (4194304, 0, 2048, 1), 0), True)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 380, in __call__
[rank4]:     return wrapper()
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 600, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 376, in wrapper
[rank4]:     return self.dispatch(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 297, in dispatch
[rank4]:     return kernel(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_higher_order_ops/utils.py", line 64, in inner
[rank4]:     return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_higher_order_ops/utils.py", line 37, in autograd_not_implemented_inner
[rank4]:     result = operator(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 380, in __call__
[rank4]:     return wrapper()
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 600, in _fn
[rank4]:     return fn(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 376, in wrapper
[rank4]:     return self.dispatch(
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 359, in dispatch
[rank4]:     return kernel(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_prims/rng_prims.py", line 172, in impl_backend_select
[rank4]:     return impl(op, *args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_prims/rng_prims.py", line 160, in impl_cuda
[rank4]:     return torch.cuda.get_rng_state(), op(*args, **kwargs)
[rank4]:   File "/home/jobuser/.local/lib/python3.10/site-packages/torch/_ops.py", line 667, in __call__
[rank4]:     return self_._op(*args, **kwargs)
[rank4]: RuntimeError: invalid dtype for bias - should match query's dtype
```