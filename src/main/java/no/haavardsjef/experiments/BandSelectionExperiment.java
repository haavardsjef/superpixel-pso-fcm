package no.haavardsjef.experiments;

import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.utility.DataLoader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() {
//		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
//		int numberOfBandsToSelect = 2;
//		Bounds bounds = new Bounds(0, 199);


//		SwarmPopulation swarmPopulation = new SwarmPopulation(50, numberOfBandsToSelect, bounds, objectiveFunction);
//		Particle solution = swarmPopulation.optimize(20, 0.5f, 0.5f, 0.2f, false);

//		List<Integer> selectedBands = solution.getDiscretePositionSorted();
		List<Integer> selectedBands = new ArrayList<>(Arrays.asList(8, 30, 41, 59, 74, 101, 111, 118, 155, 163, 176, 186));
		System.out.println("Selected bands:" + selectedBands);

		SVMClassifier svmClassifier = new SVMClassifier(new DataLoader());
		svmClassifier.evaluate(selectedBands);


	}


	public static void main(String[] args) {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();
	}


}
