sample_size = 2

split_list = [50] * (sample_size//(50)) + [i for i in [sample_size%50] if i != 0]
print(split_list)