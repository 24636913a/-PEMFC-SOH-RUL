# Determinism compatibility

{
  "cuda_strict_interpolation": {
    "synthetic_only": true,
    "device": "cuda",
    "kind": "interpolation",
    "forward_pass": true,
    "backward_pass": false,
    "optimizer_step_performed": false,
    "exception": "RuntimeError(\"upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.\")",
    "traceback": "Traceback (most recent call last):\n  File \"E:\\\u6c88\u519c\u4fe1\u7535\\ PEMFC-SOH+RUL\\RENew-PEMFC-2300h-Pytorch\\Phase_H_Determinism_MultiSeed_Stability\\scripts\\probe_determinism.py\", line 35, in strict_probe\n    try:loss.backward()\n        ^^^^^^^^^^^^^^^\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\_tensor.py\", line 626, in backward\n    torch.autograd.backward(\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\autograd\\__init__.py\", line 347, in backward\n    _engine_run_backward(\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\autograd\\graph.py\", line 823, in _engine_run_backward\n    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.\n",
    "input_shape": [
      2,
      1,
      32
    ],
    "output_shape": [
      2,
      1,
      64
    ],
    "loss": 1.4417330026626587,
    "input_sha256": "1d7cc6d02fc1fe2e21e402dc14e45c48fb2e8bfa017870b74e64b0e58e1b1954",
    "output_sha256": "6f9b34618570ee363d0185f24458f51baaa66baa9c075103367278545eac80fe",
    "status": "CUDA_STRICT_DETERMINISM_UNSUPPORTED",
    "environment": {
      "python": "3.11.9",
      "torch": "2.6.0+cu124",
      "pytorch_forecasting": "1.8.0",
      "CUDA": "12.4",
      "cuDNN": 90100,
      "GPU": "NVIDIA GeForce RTX 3070 Laptop GPU",
      "CPU_threads": 1,
      "cudnn_deterministic": true,
      "cudnn_benchmark": false,
      "torch_deterministic_algorithms": true,
      "warn_only": false,
      "TF32_matmul": false,
      "TF32_cudnn": true,
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
    }
  },
  "cuda_strict_tft": {
    "synthetic_only": true,
    "device": "cuda",
    "kind": "tft",
    "forward_pass": true,
    "backward_pass": false,
    "optimizer_step_performed": false,
    "exception": "RuntimeError(\"upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.\")",
    "traceback": "Traceback (most recent call last):\n  File \"E:\\\u6c88\u519c\u4fe1\u7535\\ PEMFC-SOH+RUL\\RENew-PEMFC-2300h-Pytorch\\Phase_H_Determinism_MultiSeed_Stability\\scripts\\probe_determinism.py\", line 35, in strict_probe\n    try:loss.backward()\n        ^^^^^^^^^^^^^^^\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\_tensor.py\", line 626, in backward\n    torch.autograd.backward(\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\autograd\\__init__.py\", line 347, in backward\n    _engine_run_backward(\n  File \"D:\\ruanjian\\python3.11.9\\Lib\\site-packages\\torch\\autograd\\graph.py\", line 823, in _engine_run_backward\n    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.\n",
    "input_shape": [
      2,
      48,
      14
    ],
    "output_shape": [
      2,
      24
    ],
    "loss": 1.9989309310913086,
    "input_sha256": "3b36db43c9ef6a78a983bcee30e63c744e63a46394d62703a5b75a6aeec7ec9c",
    "output_sha256": "d4dab8e9dbb44378f7b185f5350f24e3669626731d2194dadd65963f4e6fd22a",
    "status": "CUDA_STRICT_DETERMINISM_UNSUPPORTED",
    "environment": {
      "python": "3.11.9",
      "torch": "2.6.0+cu124",
      "pytorch_forecasting": "1.8.0",
      "CUDA": "12.4",
      "cuDNN": 90100,
      "GPU": "NVIDIA GeForce RTX 3070 Laptop GPU",
      "CPU_threads": 1,
      "cudnn_deterministic": true,
      "cudnn_benchmark": false,
      "torch_deterministic_algorithms": true,
      "warn_only": false,
      "TF32_matmul": false,
      "TF32_cudnn": true,
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
    }
  },
  "cpu_strict_interpolation": {
    "synthetic_only": true,
    "device": "cpu",
    "kind": "interpolation",
    "forward_pass": true,
    "backward_pass": true,
    "optimizer_step_performed": false,
    "exception": null,
    "traceback": null,
    "input_shape": [
      2,
      1,
      32
    ],
    "output_shape": [
      2,
      1,
      64
    ],
    "loss": 1.4599896669387817,
    "input_sha256": "0b3368f38b2f09047a18c9282fab6ae696590ffc1abdb95265f8a349a05987f1",
    "output_sha256": "606c19fd9f2fa786ef1edcfab55a6e501bb27b9c90e170dbfad2e379d1ae3afc",
    "status": "PASS",
    "environment": {
      "python": "3.11.9",
      "torch": "2.6.0+cu124",
      "pytorch_forecasting": "1.8.0",
      "CUDA": "12.4",
      "cuDNN": 90100,
      "GPU": "NVIDIA GeForce RTX 3070 Laptop GPU",
      "CPU_threads": 1,
      "cudnn_deterministic": true,
      "cudnn_benchmark": false,
      "torch_deterministic_algorithms": true,
      "warn_only": false,
      "TF32_matmul": false,
      "TF32_cudnn": true,
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
    }
  },
  "cpu_strict_tft": {
    "synthetic_only": true,
    "device": "cpu",
    "kind": "tft",
    "forward_pass": true,
    "backward_pass": true,
    "optimizer_step_performed": false,
    "exception": null,
    "traceback": null,
    "input_shape": [
      2,
      48,
      14
    ],
    "output_shape": [
      2,
      24
    ],
    "loss": 1.9844704866409302,
    "input_sha256": "5cbc0d1a64cf08c05750a1fa0b3fdf29ae9238e44854a8906b55238c5320af48",
    "output_sha256": "3ab166ff95bcd5c938be466a14f958d646e000381527c7bfeb3c9bbfc895a08a",
    "status": "PASS",
    "environment": {
      "python": "3.11.9",
      "torch": "2.6.0+cu124",
      "pytorch_forecasting": "1.8.0",
      "CUDA": "12.4",
      "cuDNN": 90100,
      "GPU": "NVIDIA GeForce RTX 3070 Laptop GPU",
      "CPU_threads": 1,
      "cudnn_deterministic": true,
      "cudnn_benchmark": false,
      "torch_deterministic_algorithms": true,
      "warn_only": false,
      "TF32_matmul": false,
      "TF32_cudnn": true,
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
    }
  }
}

Strict CUDA interpolation failure is a framework compatibility limitation, not proof of historical E2/F0 divergence cause. CPU strict success does not imply CUDA bitwise determinism. Normal training uses deterministic_algorithms=False and does not claim strict determinism. Crash-safe injection and F0/F1/F2 normal CUDA smoke evidence are separately saved.

Count erratum: state_dict_element_count=332955 includes persistent buffers, whereas trainable_parameter_count=323483 counts only optimizable F0 parameters. Phase G JSON is retained unchanged.
