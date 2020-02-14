
def compositions(number, elements):
    assert number > elements > 1
    to_process = [[i] for i in range(1, number+1)]

    while to_process:
        item = to_process.pop()
        i_sum = sum(item)
        length = len(item)

        for i in range(1, number-i_sum+1):
            newsum = i_sum+i
            if newsum <= number:
                newlist = list(item)
                newlist.append(i)
                if length == elements - 1 and newsum == number:
                    yield newlist
                elif length < elements-1 and newsum < number:
                    to_process.append(newlist)


def H1(p, q, z, t):
    if(p == q == 1):
        return 1/(1-t)
    else:
        return (H1(p-1, 1, z, t) + H1(p, q-1, z, t) - H1(p-1, q-1, z, t))/(1-z**abs(p-1)*t)

def H2(p, q, z, t, g1, g2):
    if(p == q == 1):
        return 1/(1-t*g1*g2)
    else:
        return (H1(p-1, 1, z, t) + H1(p, q-1, z, t) - H1(p-1, q-1, z, t))/(1-z**abs(p-1)*t*g1**(p-1)*g2**(q-1))


if __name__ == '__main__':
    comps = list(compositions(10, 5))
    print(comps)
