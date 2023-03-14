import no.haavardsjef.utility.DataLoader;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DataLoaderTest {

	@Test
	public void testLoadIndianPines() {
		DataLoader dataLoader = new DataLoader();
		dataLoader.loadData();
		assertEquals(200, dataLoader.getNumberOfDataPoints()); // Check number of bands is correct
		assertEquals(145 * 145, dataLoader.getDataPoint(0).length); // Check number of pixels is correct
		assertEquals(3172, dataLoader.getDataPoint(0)[0], 0.0001); // Check first pixel value is correct
		assertEquals(2580, dataLoader.getDataPoint(0)[1], 0.0001); // Check second pixel value is correct
	}

	@Test
	public void testLoadIndianPinesGroundTruth() {
		DataLoader dataLoader = new DataLoader();
		dataLoader.loadGroundTruth();
		int[] groundTruth = dataLoader.getGroundTruthFlattened();

		assertEquals(145 * 145, groundTruth.length); // Check number of pixels is correct
		assertEquals(3, groundTruth[0]); // Check first pixel value is correct
		assertEquals(3, groundTruth[1]); // Check second pixel value is correct
		assertEquals(0, groundTruth[20]); // Check 21st pixel value is correct
		assertEquals(3, groundTruth[145]); // Check first pixel on 2nd row is correct

	}

}
