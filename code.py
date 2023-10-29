def last_person_with_direction(n, m):
    seq = list(range(1, n + 1))
    idx = 0

    while len(seq) > 1:
        count = 1
        while count < m:
            idx = (idx + 1) % len(seq)
            count += 1

        seq.pop(idx)
        idx = idx % len(seq)

    return seq[0]


n = 5
m = 3
result = last_person_with_direction(n, m)
print(result)