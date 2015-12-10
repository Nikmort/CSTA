import numpy as np
import numpy.random as nrand
import matplotlib.pyplot as plt

def Latin_hypercube(ranges, sample_size):
	dimension = len(ranges)
	sample = np.zeros((sample_size, dimension))
	for i in range(dimension):
		sample[:,i] = ranges[i,0] + ((ranges[i,1] - ranges[i,0]) / sample_size) * (nrand.permutation(sample_size) + 1.0/2)
	return sample

def Normalize_vector(sample_vector):
	return (sample_vector - np.mean(sample_vector))/np.std(sample_vector)

def Heavyside(a):
	if a < 0:
		return 0.0
	else:
		return 1.0 

def example_function(function_name):
	
	def linear_function(x):
		# 4D linear function
		return 3*x[0] + 2*x[1] + 1*x[2] + 0*x[3]

	def linear_function_with_noise(x):
		# 4D linear function with noise
		return (3*x[0] + 2*x[1] + 1*x[2] + 0*x[3])*(1.0 + nrand.random_sample()) + 3*nrand.random_sample()
	
	def Sobol_function(x):
		# 6D Sobol function
		answer = 1.0
		a = [0.0, 0.5, 3.0, 9.0, 99.0, 99.0]
		for i in range(6):
			answer *= (abs(4*x[i]-2)+a[i])/(1+a[i])
		return answer

	def Sobol_function_with_noise(x):
		# 6D Sobol function
		answer = 1.0
		a = [0.0, 0.5, 3.0, 9.0, 99.0, 99.0]
		for i in range(6):
			answer *= (abs(4*x[i]-2)+a[i])/(1+a[i])
		return answer * (1.0 + 0.5 * (nrand.random_sample()-0.5)) + 2 * (nrand.random_sample()-0.5)

	if function_name == 'linear':
		return linear_function
	if function_name == 'linear_noise':
		return linear_function_with_noise
	elif function_name == 'Sobol':
		return Sobol_function
	elif function_name == 'Sobol_noise':
		return Sobol_function_with_noise

def CSTA(function_name, ranges, budget, noise=False):

	if noise == True:
		ranges = np.append(ranges, [[0.0, 1.0]], axis=0)

	dimension = len(ranges)

	function = example_function(function_name)

	sample_size = budget // (2 * (dimension + 1))

	input_prime_set = Latin_hypercube(ranges, sample_size)
	input_unprime_set = Latin_hypercube(ranges, sample_size)

	input_prime_sets = np.tile(input_prime_set, (dimension+1, 1, 1))
	input_unprime_sets = np.tile(input_unprime_set, (dimension+1, 1, 1))

	for i in range(dimension):
		input_prime_sets[i+1, :, i] = input_unprime_set[:, i]
		input_unprime_sets[i+1, :, i] = input_prime_set[:, i]

	output_prime_sets = np.zeros((dimension+1, sample_size))
	output_unprime_sets = np.zeros((dimension+1, sample_size))

	for i in range(dimension+1):
		for j in range(sample_size):
			output_prime_sets[i, j] = function(input_prime_sets[i, j])
			output_unprime_sets[i, j] = function(input_unprime_sets[i, j])
		output_prime_sets[i] = Normalize_vector(output_prime_sets[i])
		output_unprime_sets[i] = Normalize_vector(output_unprime_sets[i])

	sobol_main = np.zeros((dimension))
	sobol_total = np.zeros((dimension))
	correlation_main = np.zeros((dimension))
	correlation_total = np.zeros((dimension))
	spurious_correlation = np.zeros((dimension))
	for i in range(dimension):
		correlation_main[i] = (np.dot(output_prime_sets[0], output_unprime_sets[i+1]) + np.dot(output_unprime_sets[0], output_prime_sets[i+1])) / (2 * sample_size)
		correlation_total[i] = (np.dot(output_prime_sets[0], output_prime_sets[i+1]) + np.dot(output_unprime_sets[0], output_unprime_sets[i+1])) / (2 * sample_size)
		spurious_correlation[i] = (np.dot(output_prime_sets[0], output_unprime_sets[0]) + np.dot(output_unprime_sets[i+1], output_prime_sets[i+1])) / (2 * sample_size)
		sobol_main[i] = correlation_main[i] - Heavyside(correlation_total[i] - 0.5) * spurious_correlation[i]
		sobol_total[i] = 1 - correlation_total[i] + Heavyside(correlation_main[i] - 0.5) * spurious_correlation[i]

	if noise == False:
		return np.maximum(0, sobol_main), np.maximum(0, sobol_total)
	elif noise == True:
		return np.maximum(0, sobol_main[:(dimension-1)] / (1.0-sobol_total[dimension-1])), np.maximum(0, (sobol_total[:(dimension-1)]-sobol_total[dimension-1]) / (1.0-sobol_total[dimension-1]))

budget = 20000
exp_size = 200

def get_ranges_and_answers(function_name):
	if function_name[:5] == 'Sobol':
		ranges = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
		true_main = [0.586781, 0.260792, 0.0366738, 0.00586781, 0.00005868, 0.00005868]
		true_total = [0.690086, 0.356173, 0.0563335, 0.00917058, 0.00009201, 0.00009201]
	elif function_name[:5] == 'linea':
		ranges = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
		true_main = [9.0/14, 4.0/14, 1.0/14, 0.0]
		true_total = [9.0/14, 4.0/14, 1.0/14, 0.0]
	return len(ranges), ranges, true_main, true_total

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
axes[0, 0].set_ylabel('Main')
axes[1, 0].set_ylabel('Total')

function_type = 'linear'

if function_type == 'Sobol':
	xrow = [1, 2, 3, 4, 5, 6]
	scale = [-0.1, 1.0]
elif function_type == 'linear':
	xrow = [1, 2, 3, 4]
	scale = [-0.1, 1.0]

function_name = function_type
dimension, ranges, true_main, true_total = get_ranges_and_answers(function_name)
sobol_main = np.zeros((exp_size, dimension))
sobol_total = np.zeros((exp_size, dimension))
for i in range(exp_size):
	sobol_main[i,:], sobol_total[i,:] = CSTA(function_name, ranges, budget, noise=False)

axes[0, 0].boxplot(sobol_main, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[0, 0].scatter(xrow, true_main, s=16, c='red')
axes[0, 0].set_title('Sobol indices for \nlinear function', fontsize=12)
axes[0, 0].set_ylim(scale)

axes[1, 0].boxplot(sobol_total, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[1, 0].scatter(xrow, true_total, s=16, c='red')
axes[1, 0].set_ylim(scale)

function_name = function_type + '_noise'
dimension, ranges, true_main, true_total = get_ranges_and_answers(function_name)
sobol_main = np.zeros((exp_size, dimension))
sobol_total = np.zeros((exp_size, dimension))
for i in range(exp_size):
	sobol_main[i,:], sobol_total[i,:] = CSTA(function_name, ranges, budget, noise=False)

axes[0, 1].boxplot(sobol_main, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[0, 1].scatter(xrow, true_main, s=16, c='red')
axes[0, 1].set_title('Sobol indices for \nlinear function with noise', fontsize=12)
axes[0, 1].set_ylim(scale)

axes[1, 1].boxplot(sobol_total, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[1, 1].scatter(xrow, true_total, s=16, c='red')
axes[1, 1].set_ylim(scale)

function_name = function_type + '_noise'
dimension, ranges, true_main, true_total = get_ranges_and_answers(function_name)
sobol_main = np.zeros((exp_size, dimension))
sobol_total = np.zeros((exp_size, dimension))
for i in range(exp_size):
	sobol_main[i,:], sobol_total[i,:] = CSTA(function_name, ranges, budget, noise=True)
# if function_type == 'Sobol':
# 	sobol_main, sobol_total, true_main, true_total = np.log(sobol_main), np.log(sobol_total), np.log(true_main), np.log(true_total)

axes[0, 2].boxplot(sobol_main, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[0, 2].scatter(xrow, true_main, s=16, c='red')
axes[0, 2].set_title('Corrected Sobol indices for \nlinear function with noise', fontsize=12)
axes[0, 2].set_ylim(scale)

axes[1, 2].boxplot(sobol_total, widths = 0.8, showfliers=False, whiskerprops={'linestyle':'solid', 'color':'blue'}, medianprops={'linestyle': ''}, boxprops={'color':'blue'}, capprops={'color':'blue'})
axes[1, 2].scatter(xrow, true_total, s=16, c='red')
axes[1, 2].set_ylim(scale)

plt.savefig(function_type+str(budget)+'.png')
#plt.show()
# print(sobol_main)
# print(sobol_total)
# np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
# print(function_name)
# print('main', sobol_main)
# print('corrected_main', sobol_main/(1.0-sobol_total[dimension]))
# print('true main', true_main)
# print('total', sobol_total)
# print('corrected_total', (sobol_total-sobol_total[dimension])/(1.0-sobol_total[dimension]))
# print('true total', true_total)
