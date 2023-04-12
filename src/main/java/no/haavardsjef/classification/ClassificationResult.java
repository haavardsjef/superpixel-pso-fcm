package no.haavardsjef.classification;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.List;

public class ClassificationResult {
	private final int numClasses; // Number of classes, not counting the background class
	private final List<List<Prediction>> runs;

	/**
	 * @param numClasses Number of classes, not counting the background class
	 */
	public ClassificationResult(int numClasses) {
		this.numClasses = numClasses;
		runs = new ArrayList<>();
	}

	public void addRun(List<Prediction> predictions) {
		runs.add(predictions);
	}


	/**
	 * Gets the overall accuracy for each run
	 *
	 * @return A descriptive statistics object containing the overall accuracy for each run
	 */
	public DescriptiveStatistics getOverallAccuracy() {
		DescriptiveStatistics stats = new DescriptiveStatistics();
		for (List<Prediction> run : runs) {
			int correct = 0;
			for (Prediction pred : run) {
				if (pred.predictedLabel() == pred.trueLabel()) {
					correct += 1;
				}
			}
			double acc = (double) correct / run.size();
			stats.addValue(acc);
		}
		return stats;
	}

	/**
	 * Gets the overall accuracy for each class
	 *
	 * @return A descriptive statistics object containing the overall accuracy for each class
	 */
	public DescriptiveStatistics getAverageOverallAccuracy() {
		DescriptiveStatistics stats = new DescriptiveStatistics();
		int[] trueLabels = new int[numClasses];
		int[] correctlyPredicted = new int[numClasses];

		for (List<Prediction> run : runs) {
			for (Prediction pred : run) {
				trueLabels[pred.trueLabel() - 1] += 1;
				if (pred.trueLabel() == pred.predictedLabel()) {
					correctlyPredicted[pred.trueLabel() - 1] += 1;
				}
			}
		}

		for (int i = 0; i < numClasses; i++) {
			double acc = (double) correctlyPredicted[i] / trueLabels[i];
			stats.addValue(acc);
		}


		return stats;
	}

}
