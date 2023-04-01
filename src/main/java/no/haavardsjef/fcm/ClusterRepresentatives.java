package no.haavardsjef.fcm;

import no.haavardsjef.dataset.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Class providing methods to select cluster representatives given a set of cluster centers
 */
public class ClusterRepresentatives {

	private final Dataset dataset;
	private final List<List<Integer>> clusters;

	public ClusterRepresentatives(Dataset dataset) {
		this.dataset = dataset;
		clusters = new ArrayList<>();
	}

	public void hardClusterBands(List<Integer> clusterCentroids) {
		int numClusters = clusterCentroids.size();

		for (int i = 0; i < numClusters; i++) {
			clusters.add(new ArrayList<>());
		}

		// For each band, find the closest cluster centroid and add it to the cluster
		for (int i = 0; i < dataset.getNumBands(); i++) {
			int closestCentroid = 0;
			double closestDistance = Double.MAX_VALUE;

			for (int j = 0; j < numClusters; j++) {
				double distance = dataset.euclideanDistance(i, clusterCentroids.get(j));
				if (distance < closestDistance) {
					closestCentroid = j;
					closestDistance = distance;
				}
			}

			clusters.get(closestCentroid).add(i);
		}
	}


	public List<Integer> centroidRepresentatives(List<Integer> clusterCentroids) {
		return clusterCentroids;
	}

	public List<Integer> medianRepresentative(List<Integer> clusterCentroids) {
		return null;
	}

	public List<Integer> highestEntropyRepresentative(List<Integer> clusterCentroids) {
		return null;
	}

	public List<Integer> highestMutualInformationRepresentative(List<Integer> clusterCentroids) {
		return null;
	}

	public List<Integer> lowestKullbackLeiblerDivergenceRepresentative(List<Integer> clusterCentroids) {
		return null;
	}


}
