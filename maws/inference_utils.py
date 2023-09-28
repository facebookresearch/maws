import contextlib
import torch


_STACK = contextlib.ExitStack()


def start_inference_mode(device="cpu"):
    global _STACK
    _STACK.enter_context(torch.inference_mode())
    # _STACK.enter_context(torch.amp.autocast(device_type=device, dtype=torch.bfloat16))


def reset_inference_mode():
    global _STACK
    _STACK.close()