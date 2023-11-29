import time
from functools import wraps

def benchmark(times=100, verbose=False):
    def decorator(func, ):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Running {func.__name__}()...")

            start_time = time.perf_counter()
            for i in range(times):
                func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"Finished in {total_time:.4f} seconds.")
            
            if verbose:
                result = func(*args, **kwargs)
                print(result)
                return result

        return wrapper
    return decorator