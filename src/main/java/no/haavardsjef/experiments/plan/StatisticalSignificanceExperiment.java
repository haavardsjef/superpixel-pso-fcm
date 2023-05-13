package no.haavardsjef.experiments.plan;

import no.haavardsjef.classification.McNemarsTest;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class StatisticalSignificanceExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {

		Dataset ds = new Dataset(DatasetName.indian_pines);

		List<Integer> selectedBands1 = Arrays.asList(22, 47, 35, 81, 175, 115, 116, 117, 99, 132);
		List<Integer> selectedBands2 = Arrays.asList(36, 67, 35, 81, 30, 94, 80, 116, 127, 141);

		SVMClassifier svm = new SVMClassifier(ds);

		for (int r = 0; r < 10; r++) {


			int[][] contingencyTable = svm.compareBandSubsets(selectedBands1, selectedBands2, 0.1, false);

			// Print the contingency table
			for (int[] row : contingencyTable) {
				for (int i : row) {
					System.out.print(i + " ");
				}
				System.out.println();
			}

			double Z = McNemarsTest.computeTestStatistic(contingencyTable);
			double pValue = McNemarsTest.computePValue(contingencyTable);

			System.out.println("Z: " + Z);
			System.out.println("P-value: " + pValue);
		}


	}

	public static void main(String[] args) {
		try {
			new StatisticalSignificanceExperiment().runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
