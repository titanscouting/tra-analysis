import analysis
import time

start = time.time()
analysis.basic_analysis("data.txt")
end = time.time()

print(end - start)
