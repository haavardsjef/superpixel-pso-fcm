package no.haavardsjef.fcm;

import no.haavardsjef.AbstractFitnessFunction;
import no.haavardsjef.fcm.distancemetrics.IDistance;
import no.haavardsjef.utility.IDataLoader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class FCM extends AbstractFitnessFunction {

	private IDataLoader dataLoader;
	private float m;
	private IDistance distanceMetric;
	private HashMap<List<Integer>, Float> fitnessCache = new HashMap<>();

	public FCM(float m, IDistance distanceMetric, IDataLoader dataLoader) {
		this.m = m;
		this.distanceMetric = distanceMetric;
		this.dataLoader = dataLoader;
		dataLoader.loadData();
	}


	private double[][] updateMembershipValues(float[] position) {
		double[][] u = new double[this.dataLoader.getNumberOfDataPoints()][position.length];
		int[] centers = new int[position.length];
		for (int i = 0; i < position.length; i++) {
			centers[i] = Math.round(position[i]);
		}
		int clusterCount = position.length;
		for (int i = 0; i < dataLoader.getNumberOfDataPoints(); i++) {
			double[] i_data = dataLoader.getDataPoint(i);
			for (int j = 0; j < clusterCount; j++) {
				if (i == centers[j]) {
					u[i][j] = 1;
					continue;
				}


				float sum = 0;
				double[] j_data = dataLoader.getDataPoint(centers[j]);
				double upper = distanceMetric.distance(i_data, j_data);
				for (int k = 0; k < clusterCount; k++) {
					double[] k_data = dataLoader.getDataPoint(centers[k]);
					double lower = distanceMetric.distance(i_data, k_data);
					sum += Math.pow((upper / lower), 2 / (this.m - 1));
				}
				u[i][j] = 1 / sum; // TODO: Cache this
			}
		}
		return u;
	}

	public float evaluate(float[] position) {
		// Evaluate the fitness of the position by FCM

		// Round the position to the nearest integer
		List<Integer> clusterCenters = new ArrayList<>();
		for (int i = 0; i < position.length; i++) {
			clusterCenters.add(Math.round(position[i]));
		}
		// Sort the cluster centers
		clusterCenters.sort(Integer::compareTo);

		// Check if the fitness has already been evaluated
		if (this.fitnessCache.containsKey(clusterCenters)) {
			System.out.println("Fitness already evaluated for cluster centers: " + Arrays.toString(position) + " with fitness: " + fitnessCache.get(clusterCenters));
			return fitnessCache.get(clusterCenters);
		}


		double[][] u = updateMembershipValues(position);

		float J = 0f;
		for (int i = 0; i < dataLoader.getNumberOfDataPoints(); i++) {
			double[] i_data = dataLoader.getDataPoint(i);
			for (int j = 0; j < position.length; j++) {
				float sum = 0;
				for (int k = 0; k < position.length; k++) {
					double[] k_data = dataLoader.getDataPoint(clusterCenters.get(k));
					sum += Math.pow(u[i][k], this.m) * distanceMetric.distance(k_data, i_data);
				}
				J += sum;
			}
		}


		fitnessCache.put(clusterCenters, J);

		System.out.println("Evaluated solution with cluster centers: " + Arrays.toString(position) + " with fitness: " + J);
		return J;


	}
}
