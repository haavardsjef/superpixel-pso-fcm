package no.haavardsjef.classification;

import no.haavardsjef.utility.DataLoader;
import no.haavardsjef.utility.IDataLoader;

import java.util.List;
import java.util.stream.Collectors;

public class SVMClassifier {

	DataLoader dataLoader;

	public SVMClassifier(DataLoader dataLoader) {
		this.dataLoader = dataLoader;
		dataLoader.loadData();
	}


	public void evaluate(List<Integer> selectedBands) {
		List<double[]> bandData = selectedBands.stream().map(band -> dataLoader.getDataPoint(band)).collect(Collectors.toList());

		System.out.println("Selected bands: " + selectedBands);
		// TODO: Consider normalizing each pixel so that all bands add to 1

		// Load ground truths
		dataLoader.loadGroundTruth();
		int[] groundTruth = dataLoader.getGroundTruthFlattened();


	}

}
