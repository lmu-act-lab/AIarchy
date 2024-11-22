from student import Student
import copy

x = Student({"SES": 0}, weights={"health": 0.15, "social": 0.05, "grades": 0.8})
# cpd = x.model.get_cpds("Time studying")
# example_time_step = x.time_step(
#     fixed_evidence=x.fixed_assignment, weights=x.weights, samples=x.sample_num
# )
# average_sample = example_time_step[0]
# print(cpd.variables[0])
# print(cpd.cardinality)
# print(cpd.values[0][1][1][1])
# print(average_sample)
# print(cpd)
# # arr = cpd.values
# # for variable in cpd.variables:
# #     var_val = average_sample[variable]
# #     arr = arr[var_val]

# def update_array(arr, variables, average_sample):
#     # Get the indices from average_sample
#     indices = [average_sample[variable] for variable in variables]
#     # Convert indices to tuple to use for array indexing
#     indices = tuple(indices)
#     # Update the value at the specified indices
#     arr[indices] += 0.001

# # Use the function to update the array
# update_array(cpd.values, cpd.variables, average_sample)
# # arr += 0.1
# # cpd.values[0] += 0.1
# # print(arr)
# print(cpd)

# print(example_time_step)
x.train(5)
x.display_cpts()
