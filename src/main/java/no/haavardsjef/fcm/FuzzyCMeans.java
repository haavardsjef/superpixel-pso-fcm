package no.haavardsjef.fcm;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.Dataset;
import no.haavardsjef.DatasetName;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;

@Log4j2
public class FuzzyCMeans implements IObjectiveFunction {
	private final Dataset dataset;
	private final INDArray data;
	private final double fuzziness;
	private HashMap<String, Float> fitnessCache;

	public FuzzyCMeans(Dataset dataset, double fuzziness) {
		this.dataset = dataset;
		this.data = dataset.data;
		this.fuzziness = fuzziness;
		this.fitnessCache = new HashMap<>();
	}


//	public INDArray calculateMembershipMatrix(INDArray candidateCentroids) {
//		int numDataPoints = (int) data.size(0);
//		int numClusters = (int) candidateCentroids.size(0);
//		INDArray candidateMembershipMatrix = Nd4j.zeros(numDataPoints, numClusters);
//		double epsilon = 1e-9;
//
//		IntStream.range(0, numDataPoints).parallel().forEach(i -> {
//			INDArray dataPoint = data.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
//
//			// Create a separate distances INDArray for each thread
//			INDArray distances = Nd4j.create(numClusters);
//
//			for (int j = 0; j < numClusters; j++) {
//				distances.putScalar(j, dataPoint.distance2(candidateCentroids.get(NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all())));
//			}
//			INDArray membershipDenominator = Transforms.pow(distances.reshape(1, -1).div(distances.add(epsilon)), 2.0 / (fuzziness - 1)).sum(1);
//			candidateMembershipMatrix.putRow(i, membershipDenominator.rdiv(1.0));
//		});
//
//		return candidateMembershipMatrix;
//	}
//
//
//	public double objectiveFunction(INDArray candidateCentroids) {
//		INDArray candidateMembershipMatrix = calculateMembershipMatrix(candidateCentroids);
//
//		int numDataPoints = (int) data.size(0);
//		int numClusters = (int) candidateCentroids.size(0);
//
//		return IntStream.range(0, numDataPoints).parallel().mapToDouble(i -> {
//			double sum = 0.0;
//			for (int j = 0; j < numClusters; j++) {
//				double membership = Math.pow(candidateMembershipMatrix.getDouble(i, j), fuzziness);
//				double distance = data.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).distance2(candidateCentroids.get(NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()));
//				sum += membership * distance;
//			}
//			return sum;
//		}).sum();
//	}

	public INDArray calculateMembershipMatrix(List<Integer> candidateCentroids) {
		int numDataPoints = (int) data.size(0);
		int numClusters = candidateCentroids.size();
		INDArray candidateMembershipMatrix = Nd4j.zeros(numDataPoints, numClusters);
		double epsilon = 1e-9;

		IntStream.range(0, numDataPoints).parallel().forEach(i -> {
			INDArray distances = Nd4j.create(numClusters);
			for (int j = 0; j < numClusters; j++) {
				distances.putScalar(j, dataset.euclideanDistance(i, candidateCentroids.get(j)));
			}
			INDArray membershipDenominator = Transforms.pow(distances.reshape(1, -1).div(distances.add(epsilon)), 2.0 / (fuzziness - 1)).sum(1);
			candidateMembershipMatrix.putRow(i, membershipDenominator.rdiv(1.0));
		});

		return candidateMembershipMatrix;
	}

	public double objectiveFunction(List<Integer> candidateCentroids) {
		INDArray candidateMembershipMatrix = calculateMembershipMatrix(candidateCentroids);

		int numDataPoints = (int) data.size(0);
		int numClusters = candidateCentroids.size();

		AtomicReferenceArray<Double> atomicArray = new AtomicReferenceArray<>(numDataPoints * numClusters);
		IntStream.range(0, numDataPoints).parallel().forEach(i -> {
			IntStream.range(0, numClusters).parallel().forEach(j -> {
				double membership = Math.pow(candidateMembershipMatrix.getDouble(i, j), fuzziness);
				double distance = dataset.euclideanDistance(i, candidateCentroids.get(j));
				double product = membership * distance;
				atomicArray.set(i * numClusters + j, product);
			});
		});

		double sum = 0.0;
		for (int i = 0; i < atomicArray.length(); i++) {
			sum += atomicArray.get(i);
		}

		return sum;
	}


	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset("data/indian_pines", DatasetName.indian_pines);
		double fuzziness = 2.0;

		FuzzyCMeans fuzzyCMeans = new FuzzyCMeans(ds, fuzziness);


		// Get 10 bands
		List<Integer> bandList = new ArrayList<>();
		for (int i = 0; i < 100; i++) {
			bandList.add(i);
		}

		fuzzyCMeans.evaluate(bandList);

	}

	@Override
	public float evaluate(List<Integer> candidateSolution) {
		long startTime = System.currentTimeMillis();
		float result = 0.0f;
		// Check if we have already evaluated this solution
		if (this.fitnessCache.containsKey(candidateSolution.toString())) {
			log.info("Found solution in cache. Returning cached fitness value.");
			result = this.fitnessCache.get(candidateSolution.toString());
		} else {
			result = (float) this.objectiveFunction(candidateSolution);

			// Add to cache
			this.fitnessCache.put(candidateSolution.toString(), result);
		}

		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		log.info("Evaluated solution with cluster centers: " + Arrays.toString(candidateSolution.toArray()) + " with fitness: " + result + " in " + elapsedTime + "ms");
		return result;
	}
}

