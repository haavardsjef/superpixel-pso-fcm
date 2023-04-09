package no.haavardsjef.classification;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.List;

public interface IClassifier {

	public DescriptiveStatistics evaluate(List<Integer> selectedBands, int numClassificationRuns); // TODO: Make this return some kind of result object
}
