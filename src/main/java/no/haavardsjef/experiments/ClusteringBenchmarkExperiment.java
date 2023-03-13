package no.haavardsjef.experiments;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoaderCSV;
import no.haavardsjef.vizualisation.PlotScatter;

import java.util.List;
import java.util.ArrayList;

public class ClusteringBenchmarkExperiment implements IExperiment {

	public void runExperiment() {

		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoaderCSV());
		Bounds bounds = new Bounds(0, 999);
		SwarmPopulation swarmPopulation = new SwarmPopulation(1000, 2, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false);


		List<Integer> clusterCenterIndexes = solution.getDiscretePositionSorted();

		// Plot the clusters
		PlotScatter plotScatter = new PlotScatter();
		DataLoaderCSV dataLoader = new DataLoaderCSV();
		dataLoader.loadData();
		plotScatter.plot(dataLoader.getData(), clusterCenterIndexes);
	}


}
