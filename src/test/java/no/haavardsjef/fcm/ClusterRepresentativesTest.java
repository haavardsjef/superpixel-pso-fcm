package no.haavardsjef.fcm;

import junit.framework.TestCase;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;

import java.io.IOException;
import java.util.List;

@Log4j2
public class ClusterRepresentativesTest extends TestCase {

	public void testHardClusterBands() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);


		List<Integer> clusterCentroids = List.of(100);
		ClusterRepresentatives clusterRepresentatives = new ClusterRepresentatives(dataset);
		clusterRepresentatives.hardClusterBands(clusterCentroids);
		List<Integer> representatives = clusterRepresentatives.centroidRepresentatives(clusterCentroids);
		log.info("Centroid representative: {}", representatives);
		assertEquals(clusterCentroids, representatives);

		representatives = clusterRepresentatives.meanRepresentative(clusterCentroids);
		log.info("Mean representative: {}", representatives);

		representatives = clusterRepresentatives.highestEntropyRepresentative(clusterCentroids);
		log.info("Highest entropy representative: {}", representatives);

	}
}