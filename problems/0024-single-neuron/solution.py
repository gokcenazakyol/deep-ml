import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	probabilities = []
	for sample in features:
		z = sum((w*x) for w,x in zip(weights, sample)) + bias
		prob= 1 / (1 + math.exp(-z))
		probabilities.append(round(prob,4))

	n = len(probabilities)
	mse = sum((p-y)**2 for p,y in zip(probabilities,labels)) / n
	mse = round(mse,4)
	return probabilities, mse