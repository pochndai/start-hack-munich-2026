"""A noisy neighbour: keep allocating, touching and freeing memory blocks of mixed sizes until killed."""
import random, sys, numpy as np
total_gb = float(sys.argv[1]); held, size = [], 0
while True:
    n = random.choice([3, 5, 17, 65, 257, 1025]) * 2**20  # odd sizes in MB, so freed holes do not line up with large pages
    held.append(np.ones(n // 8)); size += n
    while size > total_gb * 2**30:
        size -= held.pop(random.randrange(len(held))).nbytes
