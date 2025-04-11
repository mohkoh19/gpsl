import time


# Gets a func an a list of args. Runs a for loop calling func(arg) for each arg in args.
# Measures time for each iteration and returns the max time
# and the results of all the calls to func.
def max_iteration_time(func, args, tt=0.0, mt=0.0):
    max_time = 0
    total_start = time.time()
    results = []
    for arg in args:
        start = time.time()

        # If arg is a tuple, we unpack it
        if isinstance(arg, tuple):
            result = func(*arg)
        else:
            result = func(arg)

        end = time.time()
        max_time = max(max_time, end - start)
        results.append(result)
    total_end = time.time()
    total_time = total_end - total_start
    return tt + total_time, mt + max_time, results


def client_backward(client_output):
    client_output.backward(
        client_output,
        retain_graph=True,
    )
