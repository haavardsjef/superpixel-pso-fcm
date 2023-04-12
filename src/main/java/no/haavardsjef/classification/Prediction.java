package no.haavardsjef.classification;


import java.io.Serializable;

public record Prediction(int pixelIndex, int trueLabel, int predictedLabel) {
}
