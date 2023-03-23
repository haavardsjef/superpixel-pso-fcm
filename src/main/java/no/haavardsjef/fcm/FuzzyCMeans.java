package no.haavardsjef.fcm;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.Dataset;
import no.haavardsjef.DatasetName;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.stream.IntStream;

@Log4j2
public class FuzzyCMeans {
	private INDArray data;
	private double fuzziness;

	public FuzzyCMeans(INDArray data, double fuzziness) {
		this.data = data;
		this.fuzziness = fuzziness;
	}


	public INDArray calculateMembershipMatrix(INDArray candidateCentroids) {
		int numDataPoints = (int) data.size(0);
		int numClusters = (int) candidateCentroids.size(0);
		INDArray candidateMembershipMatrix = Nd4j.zeros(numDataPoints, numClusters);
		double epsilon = 1e-9;

		IntStream.range(0, numDataPoints).parallel().forEach(i -> {
			INDArray dataPoint = data.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());

			// Create a separate distances INDArray for each thread
			INDArray distances = Nd4j.create(numClusters);

			for (int j = 0; j < numClusters; j++) {
				distances.putScalar(j, dataPoint.distance2(candidateCentroids.get(NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all())));
			}
			INDArray membershipDenominator = Transforms.pow(distances.reshape(1, -1).div(distances.add(epsilon)), 2.0 / (fuzziness - 1)).sum(1);
			candidateMembershipMatrix.putRow(i, membershipDenominator.rdiv(1.0));
		});

		return candidateMembershipMatrix;
	}


	public double objectiveFunction(INDArray candidateCentroids) {
		INDArray candidateMembershipMatrix = calculateMembershipMatrix(candidateCentroids);

		int numDataPoints = (int) data.size(0);
		int numClusters = (int) candidateCentroids.size(0);

		return IntStream.range(0, numDataPoints).parallel().mapToDouble(i -> {
			double sum = 0.0;
			for (int j = 0; j < numClusters; j++) {
				double membership = Math.pow(candidateMembershipMatrix.getDouble(i, j), fuzziness);
				double distance = data.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).distance2(candidateCentroids.get(NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()));
				sum += membership * distance;
			}
			return sum;
		}).sum();
	}


	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset("data/indian_pines", DatasetName.indian_pines);
		INDArray data = ds.data;

		double fuzziness = 2.0;

		FuzzyCMeans fuzzyCMeans = new FuzzyCMeans(data, fuzziness);


		// Get 10 bands
		int[] bands = {9, 11, 69, 70, 76, 82, 90, 95, 101, 115};
		INDArray band1INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[0]), NDArrayIndex.all());
		INDArray band2INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[1]), NDArrayIndex.all());
		INDArray band3INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[2]), NDArrayIndex.all());
		INDArray band4INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[3]), NDArrayIndex.all());
		INDArray band5INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[4]), NDArrayIndex.all());
		INDArray band6INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[5]), NDArrayIndex.all());
		INDArray band7INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[6]), NDArrayIndex.all());
		INDArray band8INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[7]), NDArrayIndex.all());
		INDArray band9INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[8]), NDArrayIndex.all());
		INDArray band10INDArray = data.get(NDArrayIndex.all(), NDArrayIndex.point(bands[9]), NDArrayIndex.all());

		INDArray candidateCentroids = Nd4j.stack(0, band1INDArray, band2INDArray, band3INDArray, band4INDArray, band5INDArray, band6INDArray, band7INDArray, band8INDArray, band9INDArray, band10INDArray);
		log.info("Calculating objective function value for the candidate solution");
		// Start timer
		long startTime = System.currentTimeMillis();
		double objectiveFunctionValue = fuzzyCMeans.objectiveFunction(candidateCentroids);
		// Stop timer
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		log.info("Evaluation time: " + elapsedTime + " ms");
		System.out.println("Objective function value for the candidate solution: " + objectiveFunctionValue);

	}
}

