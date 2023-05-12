package no.haavardsjef.dataset;

import junit.framework.TestCase;

import java.io.IOException;

public class DatasetTest extends TestCase {

	public void testCalculateProbabilityDistributionsSP() throws IOException {

		Dataset ds = new Dataset(DatasetName.Pavia);
		ds.setupSuperpixelContainer(900, 10000f);

		ds.calculateProbabilityDistributionsSP();

		double[][][] probDistributionsSP = ds.getProbabilityDistributionsSP();
		for (int i = 0; i < probDistributionsSP.length; i++) {
			for (int j = 0; j < probDistributionsSP[i].length; j++) {
				double sum = 0;
				for (int k = 0; k < probDistributionsSP[i][j].length; k++) {
					sum += probDistributionsSP[i][j][k];
				}
				// Check that the probabilities sum up to 1 (or very close due to floating point precision)
				assertEquals(1.0, sum, 0.001);
			}
		}
	}
}