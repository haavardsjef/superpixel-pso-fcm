package no.haavardsjef;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;
import no.haavardsjef.utility.DataLoaderCSV;
import no.haavardsjef.vizualisation.PlotScatter;
import no.haavardsjef.utility.DataLoader;

import java.util.List;
import java.util.ArrayList;

public class ClusteringBenchmarkExperiment {

	public void runExperiment() {

		AbstractFitnessFunction fitnessFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoaderCSV());
		Bounds bounds = new Bounds(0, 999)
		SwarmPopulation swarmPopulation = new SwarmPopulation(1000, 2, bounds, fitnessFunction);
		float[] solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);
		List<Integer> clusterCenterIndexes = new ArrayList<>();

		for (int i = 0; i < solution.length; i++) {
			// Round to nearest integer
			clusterCenterIndexes.add(Math.round(solution[i]));
		}

		// Plot the clusters
		PlotScatter plotScatter = new PlotScatter();
		DataLoaderCSV dataLoader = new DataLoaderCSV();
		dataLoader.loadData();
		plotScatter.plot(dataLoader.getData(), clusterCenterIndexes);
	}


}
