def Simple_number_finding(input_number):
    input_number = int(input_number)
    output_list  = []

    while (input_number % 2 == 0):
        output_list.append(2)
        input_number = input_number // 2

    while (input_number % 3 == 0):
        output_list.append(3)
        input_number = input_number // 3

    while (input_number % 5 == 0):
        output_list.append(5)
        input_number = input_number // 5

    if input_number == 1:
        return output_list
    else:
        return("None")

print("Please input a number for check!")
a = input("input_number:")
print(Simple_number_finding(a))
