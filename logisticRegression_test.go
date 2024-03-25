package goml

import (
	"math"
	"testing"
)

func TestPredictEmptyFeatures(t *testing.T) {
	lr := LogisticRegression{
		Weights: []float64{0.5, 0.5},
		Bias:    0.1,
	}
	features := []float64{}

	expected := 0.5
	result := lr.Predict(features)

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}
func TestPredictValidInput(t *testing.T) {
	lr := LogisticRegression{
		Weights: []float64{0.5, 0.5},
		Bias:    0.1,
	}
	features := []float64{1.0, 2.0}

	expected := 0.8807970779778823
	result := lr.Predict(features)

	if math.Abs(result-expected) > 1e-9 {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}
