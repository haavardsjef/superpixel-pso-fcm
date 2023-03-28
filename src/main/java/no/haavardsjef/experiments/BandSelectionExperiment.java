package no.haavardsjef.experiments;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() throws IOException {
//		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0);
		int numberOfBandsToSelect = 10;
		Bounds bounds = dataset.getBounds();


		SwarmPopulation swarmPopulation = new SwarmPopulation(100, numberOfBandsToSelect, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);

		List<Integer> selectedBands = solution.getDiscretePositionSorted();
//		List<Integer> selectedBands = new ArrayList<>(Arrays.asList(2, 22));
		System.out.println("Selected bands:" + selectedBands);

		SVMClassifier svmClassifier = new SVMClassifier(dataset);
		svmClassifier.evaluate(selectedBands);


	}


	public static void main(String[] args) throws IOException {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();
	}


}
