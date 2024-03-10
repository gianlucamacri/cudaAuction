
#!/usr/bin/env python
if __name__ == '__main__':
    import create_config
    raise SystemExit(create_config.main())

SEED = 424242

LIMIT = 10000

FILENAME = "config_semi_sparse_reduced.txt"

DENSITY = 0.5

OBJ_MULT_FACTOR = 1

TEST_CASE_PER_DIM = 3

def evaluate_step(start, end, step_number ):
    return pow((1.0 * end) / start, 1.0/step_number)


def main():
    base_dim = 500
    target_dim = 10000
    step_number = 10

    step = evaluate_step(base_dim, target_dim, step_number-1)

    with open(FILENAME, "w") as f:
        f.write(f'{SEED} {LIMIT}\n')
        m = base_dim
        for _ in range(step_number):
            f.write(TEST_CASE_PER_DIM*f'{int(m)} {int(m*OBJ_MULT_FACTOR)} {DENSITY}\n')
            m = m*step



        
    





