package no.haavardsjef.fcm.utility;

import no.haavardsjef.dataset.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Comparator;
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

	public enum RepresentativeMethod {
		clusterCentroid,
		mean,
		highestEntropy,
		weightedSum,
		rankingHybrid,
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

	public List<Integer> selectRepresentatives(List<Integer> clusterCentroids, RepresentativeMethod representativeMethod) {
		switch (representativeMethod) {
			case clusterCentroid:
				return centroidRepresentatives(clusterCentroids);
			case mean:
				return meanRepresentative(clusterCentroids);
			case highestEntropy:
				return highestEntropyRepresentative(clusterCentroids);
			case weightedSum:
				return weightedSumRepresentative();
			case rankingHybrid:
				return rankingHybridRepresentative();
			default:
				throw new IllegalArgumentException("No representative method selected");
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
			if (cluster.size() == 0) {
				continue;
			}
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
			if (cluster.size() == 0) {
				continue;
			}
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

	/**
	 * A hybrid approach that combines the highest entropy and central tendency methods is designed to balance the benefits of both
	 * methods while addressing their individual limitations. This approach aims to select a representative band that both maximizes
	 * information content and closely resembles other bands in the cluster.
	 * To compute the weighted sum we first must compute and standardize the entropy and central tendency criterion to the same range.
	 *
	 * @return
	 */
	public List<Integer> weightedSumRepresentative() {
		List<Integer> representatives = new ArrayList<>();
		List<Double> entropies = dataset.getEntropies();
		for (List<Integer> cluster : clusters) {
			if (cluster.size() == 0) {
				continue;
			}
			// Calculate mean
			INDArray clusterBandData = dataset.getBands(cluster);
			INDArray mean = clusterBandData.mean(0);

			double highestWeightedSum = Double.NEGATIVE_INFINITY;
			int bestBand = -1;

			List<Double> clusterEntropies = new ArrayList<>();
			List<Double> clusterDistances = new ArrayList<>();

			// Get entropies and distances
			for (int i = 0; i < cluster.size(); i++) {
				double entropy = entropies.get(cluster.get(i));
				clusterEntropies.add(entropy);
				double distance = mean.distance2(dataset.getBand(cluster.get(i)));
				clusterDistances.add(distance);
			}

			// Normalize entropies and distances
			double maxEntropy = clusterEntropies.stream().mapToDouble(Double::doubleValue).max().getAsDouble();
			double minEntropy = clusterEntropies.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
			double maxDistance = clusterDistances.stream().mapToDouble(Double::doubleValue).max().getAsDouble();
			double minDistance = clusterDistances.stream().mapToDouble(Double::doubleValue).min().getAsDouble();

			List<Double> normalizedEntropies = new ArrayList<>();
			List<Double> normalizedDistances = new ArrayList<>();

			for (int i = 0; i < cluster.size(); i++) {
				double normalizedEntropy = (clusterEntropies.get(i) - minEntropy) / (maxEntropy - minEntropy);
				if (Double.isNaN(normalizedEntropy)) {
					normalizedEntropy = 1.0;
				}
				normalizedEntropies.add(normalizedEntropy);
				double normalizedDistance = (clusterDistances.get(i) - minDistance) / (maxDistance - minDistance);
				if (Double.isNaN(normalizedDistance)) {
					normalizedDistance = 1.0;
				}
				normalizedDistances.add(normalizedDistance);
			}

			// Calculate weighted sum
			for (int i = 0; i < cluster.size(); i++) {
				double weightedSum = 1.0 * normalizedEntropies.get(i) - 0.5 * normalizedDistances.get(i);
				if (weightedSum > highestWeightedSum) {
					highestWeightedSum = weightedSum;
					bestBand = i;
				}
			}

			representatives.add(cluster.get(bestBand));

		}
		return representatives;
	}


	/**
	 * The hybrid ranking criterion for selecting a representative band combines the entropy and central tendency criteria
	 * to achieve a balance between high information content and close resemblance to the central tendency of other bands
	 * in the cluster. In this approach, each band is ranked based on their entropy and distance to the mean, which serves
	 * as the chosen central tendency measure. The overall rank is computed by summing the entropy-based rank and the
	 * distance-to-the-mean-based rank. The representative band is then determined as the one with the lowest overall rank,
	 * providing a good balance between information content and representativeness. If there is a tie, the band with the
	 * highest entropy is selected.
	 *
	 * @return
	 */
	public List<Integer> rankingHybridRepresentative() {
		List<Integer> representatives = new ArrayList<>();
		List<Double> entropies = dataset.getEntropies();

		for (List<Integer> cluster : clusters) {
			if (cluster.size() == 0) {
				continue;
			}

			// Calculate mean
			INDArray clusterBandData = dataset.getBands(cluster);
			INDArray mean = clusterBandData.mean(0);

			int bestBand = -1;
			int bestRank = Integer.MAX_VALUE;
			int bestEntropyRank = Integer.MAX_VALUE;

			List<Integer> entropyRanks = new ArrayList<>();
			List<Integer> distanceRanks = new ArrayList<>();

			// Calculate entropy ranks and distance ranks
			cluster.sort(Comparator.comparingDouble(entropies::get));
			for (int i = 0; i < cluster.size(); i++) {
				entropyRanks.add(i, cluster.get(i));
			}

			cluster.sort(Comparator.comparingDouble(band -> mean.distance2(dataset.getBand(band))));
			for (int i = 0; i < cluster.size(); i++) {
				distanceRanks.add(i, cluster.get(i));
			}

			// Calculate overall rank and find the representative band
			for (int band : cluster) {
				int entropyRank = entropyRanks.indexOf(band);
				int distanceRank = distanceRanks.indexOf(band);
				int overallRank = entropyRank + distanceRank;

				if (overallRank < bestRank) {
					bestRank = overallRank;
					bestBand = band;
					bestEntropyRank = entropyRank;
				} else if (overallRank == bestRank) {
					if (entropyRank < bestEntropyRank) {
						bestBand = band;
						bestEntropyRank = entropyRank;
					}
				}
			}

			representatives.add(bestBand);
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
