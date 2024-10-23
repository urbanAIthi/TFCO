import pickle
import time

start_time = time.time()

with open('data/i3040_newvector_am/sequence10_i3040_newvector_am_20.pkl', 'rb') as f:
    data = pickle.load(f)

end_time = time.time()

print(f'Time taken: {end_time - start_time} seconds')
