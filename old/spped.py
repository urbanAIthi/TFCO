import torch
import time

'''
def intensive_operations():
    #device = torch.device('cuda:0')
    device = torch.device('cpu')
    # Create large random tensors
    t_init = time.time()
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    print(f'Creating tensors took {time.time() - t_init} seconds')
    
    # Perform matrix multiplication
    result = torch.mm(a, b)
    
    # Perform element-wise multiplication
    result = a * b
    
    # Perform addition
    result = a + b
    
    # Perform subtraction
    result = a - b
    
    # Perform division
    result = a / b
    
    return result

def main():
    start_time = time.time()
    
    # Number of iterations
    n_iterations = 10
    
    for i in range(n_iterations):
        print(f"Iteration {i + 1}")
        intensive_operations()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total time taken: {elapsed_time} seconds')

if __name__ == "__main__":
    main()
'''

import time
import numpy as np
import multiprocessing

def cpu_intensive_task(n):
    # Create random matrices
    mat_a = np.random.rand(n, n)
    mat_b = np.random.rand(n, n)
    # Perform matrix multiplication
    result = np.dot(mat_a, mat_b)
    return result

def run_benchmark(n, parallel=False):
    tasks = 4  # Assume a 4-core CPU
    start_time = time.time()
    
    if parallel:
        # Run tasks in parallel to utilize all CPU cores
        with multiprocessing.Pool() as pool:
            results = pool.map(cpu_intensive_task, [n] * tasks)
    else:
        # Run tasks serially
        results = [cpu_intensive_task(n) for _ in range(tasks)]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total time taken: {elapsed_time:.4f} seconds')

if __name__ == "__main__":
    n = 10000  # Matrix dimension, adjust as needed
    print("Running in serial:")
    run_benchmark(n, parallel=False)
    print("Running in parallel:")
    run_benchmark(n, parallel=True)