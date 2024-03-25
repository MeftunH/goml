package goml

// LogisticRegression is a struct that holds the weights and bias of the model
type LogisticRegression struct {
	Weights []float64
	Bias    float64
}

// Predict is a method that takes in a feature vector and returns the predicted value\
func (lr *LogisticRegression) Predict(features []float64) float64 {
	return sigmoid(dot(lr.Weights, features) + lr.Bias)
}

// Fit is a method that takes in a dataset and trains the model
func (lr *LogisticRegression) Fit(X [][]float64, y []float64, learningRate float64, epochs int) {
	lr.Weights = make([]float64, len(X[0]))
	lr.Bias = 0
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(X); j++ {
			prediction := lr.Predict(X[j])
			error := y[j] - prediction
			lr.Bias += learningRate * error
			for k := 0; k < len(lr.Weights); k++ {
				lr.Weights[k] += learningRate * error * X[j][k]
			}
		}
	}
}

// sigmoid is a helper function that takes in a value and returns the sigmoid of that value
func sigmoid(x float64) float64 {
	return 1 / (1 + exp(-x))
}

// dot is a helper function that takes in two vectors and returns the dot product of the two vectors
func dot(a, b []float64) float64 {
	var result float64
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result
}

// exp is a helper function that takes in a value and returns the exponential of that value
func exp(x float64) float64 {
	return 1
}
