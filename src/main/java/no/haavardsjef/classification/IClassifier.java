package no.haavardsjef.classification;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.List;

public interface IClassifier {

	ClassificationResult evaluate(List<Integer> selectedBands, int numClassificationRuns);
}
