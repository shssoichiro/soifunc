"""Internal utility functions"""


def value_error(obj1, *args, num_stacks=1):
    name = get_func_name(num_stacks + 1)
    return ValueError(f"[soifunc.{name}] {obj1}", *args)


def type_error(obj1, *args, num_stacks=1):
    name = get_func_name(num_stacks + 1)
    return TypeError(f"[soifunc.{name}] {obj1}", *args)


def get_func_name(num_of_call_stacks=1):
    import inspect

    frame = inspect.currentframe()
    for _ in range(num_of_call_stacks):
        frame = frame.f_back
    return frame.f_code.co_name
