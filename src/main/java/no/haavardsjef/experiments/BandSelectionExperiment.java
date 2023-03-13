package no.haavardsjef.experiments;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;

import java.util.ArrayList;
import java.util.List;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() {
		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		int numberOfBandsToSelect = 2;
		Bounds bounds = new Bounds(0, 199);


		SwarmPopulation swarmPopulation = new SwarmPopulation(100, numberOfBandsToSelect, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);

		List<Integer> selectedBands = solution.getDiscretePositionSorted();
		System.out.println("Selected bands:" + selectedBands);


	}


	public static void main(String[] args) {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();
	}


}
