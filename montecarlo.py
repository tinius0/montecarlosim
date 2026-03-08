import random

def run_simulation():
    
    # 1. generate random inputs
    x = random.random()
    
    # 2. simulate system
    result = x**2
    
    return result


N = 10000
results = []

for _ in range(N):
    results.append(run_simulation())

average = sum(results) / N
print(average)