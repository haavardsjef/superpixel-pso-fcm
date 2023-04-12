package no.haavardsjef.classification;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.List;

public class ClassificationResult {
	private final List<List<Prediction>> runs;

	public ClassificationResult() {
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

}
