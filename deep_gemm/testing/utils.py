import functools
import os
import torch
from typing import Callable

def get_arch_major() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major


def is_sm100_family() -> bool:
    """SM100 (B200) and SM110 (Jetson Thor) share the same kernel paths."""
    return get_arch_major() in (10, 11)


def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f'{func.__name__}:')
                print(f' > Filtered by {condition}')
                print()
        return wrapper
    return decorator


def ignore_env(name: str, condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                saved = os.environ.pop(name, None)
                func(*args, **kwargs)
                if saved is not None:
                    os.environ[name] = saved
            else:
                func(*args, **kwargs)
                
        return wrapper
    return decorator
