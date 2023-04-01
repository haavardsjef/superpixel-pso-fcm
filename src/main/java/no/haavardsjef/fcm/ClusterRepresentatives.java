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

	/**
	 * Gets the band closest to the mean of the cluster
	 *
	 * @param clusterCentroids
	 * @return the bandIndex of the bands closest to the mean of the cluster, using euclidean distance
	 */
	public List<Integer> meanRepresentative(List<Integer> clusterCentroids) {
		List<Integer> representatives = new ArrayList<>();

		for (List<Integer> cluster : clusters) {
			INDArray clusterBandData = dataset.getBands(cluster);
			INDArray mean = clusterBandData.mean(0);

			int closestBand = 0;
			double closestDistance = Double.MAX_VALUE;

			for (int i = 0; i < cluster.size(); i++) {
				double distance = mean.distance2(dataset.getBand(cluster.get(i)));
				if (distance < closestDistance) {
					closestBand = i;
					closestDistance = distance;
				}
			}
			representatives.add(cluster.get(closestBand));
		}
		return representatives;
	}

	/**
	 * Gets the band with the highest entropy in the cluster
	 *
	 * @param clusterCentroids the cluster centroids
	 * @return the bandIndex of the band with the highest entropy in the cluster
	 */
	public List<Integer> highestEntropyRepresentative(List<Integer> clusterCentroids) {
		List<Integer> representatives = new ArrayList<>();
		List<Double> entropies = dataset.getEntropies();
		for (List<Integer> cluster : clusters) {
			int highestEntropyBand = 0;
			double highestEntropy = Double.MIN_VALUE;

			for (int i = 0; i < cluster.size(); i++) {
				double entropy = entropies.get(cluster.get(i));
				if (entropy > highestEntropy) {
					highestEntropyBand = i;
					highestEntropy = entropy;
				}
			}
			representatives.add(cluster.get(highestEntropyBand));
		}
		return representatives;
	}


	public List<Integer> highestMutualInformationRepresentative(List<Integer> clusterCentroids) {
		return null;
	}

	public List<Integer> lowestKullbackLeiblerDivergenceRepresentative(List<Integer> clusterCentroids) {
		return null;
	}


}
