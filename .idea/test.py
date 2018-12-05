def binary_search(input_array, value):
    """Your code goes here."""
    idx = len(input_array)/2
    forward = false
    backward = false
    while(idx<len(input_array)):
        if(input_array[idx]==value):
            return idx
        elif(input_array[idx] > value and not back):
            idx = idx/2
            front = True
        elif  not front:
            idx = idx + idx/2
            back = True
    return -1

test_list = [1,3,9,11,15,19,29]
test_val1 = 25
test_val2 = 15
print binary_search(test_list, test_val1)
print binary_search(test_list, test_val2)
