package no.haavardsjef.fcm;

import no.haavardsjef.Dataset;

import java.util.List;

/**
 * Class providing methods to select cluster representatives given a set of cluster centers
 */
public class ClusterRepresentatives {

	private final Dataset dataset;

	public ClusterRepresentatives(Dataset dataset) {
		this.dataset = dataset;
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
