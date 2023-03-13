package no.haavardsjef.experiments;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() {
		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		Bounds bounds = new Bounds(0, 199);
		SwarmPopulation swarmPopulation = new SwarmPopulation(100, 2, bounds, objectiveFunction);
		float[] solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, true);
	}


}
