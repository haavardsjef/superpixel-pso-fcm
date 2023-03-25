package no.haavardsjef.fcm;

import junit.framework.TestCase;
import no.haavardsjef.Dataset;
import no.haavardsjef.DatasetName;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class FuzzyCMeansTest extends TestCase {

	public void testEvaluate() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);

		FuzzyCMeans fuzzyCMeans = new FuzzyCMeans(dataset, 2.0);

		List<Integer> bands = new ArrayList<Integer>();
		IntStream.range(0, 20).forEach(i -> bands.add(i));


		// Calculate the objective function using the two different methods, time both methods
		long startTime = System.currentTimeMillis();
		double result = fuzzyCMeans.objectiveFunction(bands);
		long endTime = System.currentTimeMillis();
		System.out.println("Method 1 - Time taken: " + (endTime - startTime) + "ms");

		startTime = System.currentTimeMillis();
		INDArray data = dataset.getBands(bands);
		double result2 = fuzzyCMeans.objectiveFunction(data);
		endTime = System.currentTimeMillis();
		System.out.println("Method 2 - Time taken: " + (endTime - startTime) + "ms");

		// Check that the two methods of calculating the objective function return the same result within some error margin
		assertEquals(result, result2, 0.1);

	}
}