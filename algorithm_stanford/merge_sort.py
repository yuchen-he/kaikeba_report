import numpy as np

def merge_sort(unsorted_array): 

	n = len(unsorted_array)
	sorted_array = []

	if n > 1:

		left_part  = unsorted_array[:(n//2)]
		right_part = unsorted_array[(n//2):]
		# print('left_part :\n ', left_part)
		# print('right_part:\n ', right_part)
		sorted_left  = merge_sort(left_part)
		sorted_right = merge_sort(right_part)
		# print('sorted_left :\n ', sorted_left)
		# print('sorted_right:\n ', sorted_right)

		i = 0
		j = 0

		for k in range(n):
			if i == len(sorted_left):
				sorted_array.append(sorted_right[j])
				j += 1

			elif j == len(sorted_right):
				sorted_array.append(sorted_left[i])
				i += 1

			elif (sorted_left[i] <= sorted_right[j]):
				sorted_array.append(sorted_left[i])
				i += 1
			else:
				sorted_array.append(sorted_right[j])
				j += 1

	else:
		return unsorted_array

	return sorted_array

if __name__ == '__main__':
	a = [3, 9, 5, 4, 23, 63, 34, 12, 43, 55, 10, 3, 9, 5, 4, 7, 10, 35]
	a = np.random.randn(50)
	print(sorted(a))
	print(merge_sort(a))
