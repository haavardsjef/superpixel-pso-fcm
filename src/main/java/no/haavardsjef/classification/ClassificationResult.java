package no.haavardsjef.classification;

import no.haavardsjef.Dataset;
import no.haavardsjef.DatasetName;


import java.util.List;

public class ClassificationResult {
	private DatasetName datasetName;
	private int numberOfCorrectlyClassifiedSamples;
	private int numberOfSamples;
	private List<Integer> usedBands;
	private int[][] confusionMatrix;
}
