package no.haavardsjef.experiments;

import no.haavardsjef.dataset.BenchmarkDataset;
import no.haavardsjef.dataset.BenchmarkDatasetName;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.vizualisation.PlotScatter;

import java.io.IOException;
import java.util.List;

public class ClusteringBenchmarkExperiment implements IExperiment {

	public void runExperiment() throws IOException {

		BenchmarkDataset dataset = new BenchmarkDataset(BenchmarkDatasetName.clustering_hard);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0);
		Bounds bounds = dataset.getBounds();
		SwarmPopulation swarmPopulation = new SwarmPopulation(50, 10, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, false, true);


		List<Integer> clusterCenterIndexes = solution.getDiscretePositionSorted();


		// Plot the clusters
		PlotScatter plotScatter = new PlotScatter();
		plotScatter.plot(dataset.getDataAsArray(), clusterCenterIndexes);
	}

	public static void main(String[] args) throws IOException {
		ClusteringBenchmarkExperiment clusteringBenchmarkExperiment = new ClusteringBenchmarkExperiment();
		clusteringBenchmarkExperiment.runExperiment();
	}


}
