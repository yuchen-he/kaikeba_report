def karatsuba(x, y):
    if x < 10 or y < 10:
        return x*y
    #making the two numbers strings here
    str_x = str(x)
    str_y = str(y)
    #finding the mid point for each number here
    n = max(len(str(x)), len(str(y)))
    n_2 = int(n / 2)
    #higher bits of each number
    x_h = int(str_x[:-n_2])
    y_h = int(str_y[:-n_2])
    #lower bits for each number here
    x_l = int(str_x[-n_2:])
    y_l = int(str_y[-n_2:])
    a = karatsuba(x_h, y_h)
    d = karatsuba(x_l, y_l)
    e = karatsuba(x_h + x_l, y_h + y_l) - a - d
    return a*10**len(str_x) + e*10**(len(str_x) // 2) + d
result = karatsuba(1234, 8765)
print(result)
