package no.haavardsjef.experiments;

import no.haavardsjef.AbstractFitnessFunction;
import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;
import no.haavardsjef.utility.DataLoaderCSV;

public class BandSelectionExperiment {


	public void runExperiment() {
		AbstractFitnessFunction fitnessFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		Bounds bounds = new Bounds(0, 199);
		SwarmPopulation swarmPopulation = new SwarmPopulation(100, 2, bounds, fitnessFunction);
		float[] solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);
	}


}
