package no.haavardsjef.experiments;

import no.haavardsjef.Dataset;
import no.haavardsjef.DatasetName;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;

import java.io.IOException;
import java.util.List;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() throws IOException {
//		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		Dataset dataset = new Dataset("data/indian_pines", DatasetName.indian_pines);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0);
		int numberOfBandsToSelect = 5;
		Bounds bounds = new Bounds(0, 199);


		SwarmPopulation swarmPopulation = new SwarmPopulation(100, numberOfBandsToSelect, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);

		List<Integer> selectedBands = solution.getDiscretePositionSorted();
//		List<Integer> selectedBands = new ArrayList<>(Arrays.asList(8, 30, 41, 59, 74, 101, 111, 118, 155, 163, 176, 186));
		System.out.println("Selected bands:" + selectedBands);

		SVMClassifier svmClassifier = new SVMClassifier(new DataLoader());
		svmClassifier.evaluate(selectedBands);


	}


	public static void main(String[] args) throws IOException {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();
	}


}
