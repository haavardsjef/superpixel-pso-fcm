package no.haavardsjef.classification;

import no.haavardsjef.dataset.DatasetName;


import java.util.List;

public class ClassificationResult {
	private DatasetName datasetName;
	private int numberOfCorrectlyClassifiedSamples;
	private int numberOfSamples;
	private List<Integer> usedBands;
	private int[][] confusionMatrix;
}
