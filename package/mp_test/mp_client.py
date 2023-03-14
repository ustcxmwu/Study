import random
from multiprocessing import Pool


def calculate_pi(nbr_estimates):
    n = 0
    for s in range(int(nbr_estimates)):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y <= 1:
            n += 1
    return n


def main():
    nbr_samples_in_total = 1e8
    nbr_blocks = 4
    pool = Pool(processes=nbr_blocks)
    nbr_per_process = [nbr_samples_in_total/nbr_blocks] * nbr_blocks
    nbr_result = pool.map(calculate_pi, nbr_per_process)
    pi_estimate = sum(nbr_result) * nbr_blocks / nbr_samples_in_total
    print(pi_estimate)


if __name__ == "__main__":
    main()