package no.haavardsjef.classification;

import java.io.Serializable;

public record Sample(int pixelIndex, int label, double[] features) implements Serializable {

}

