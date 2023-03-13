package no.haavardsjef.fcm;

import no.haavardsjef.fcm.distancemetrics.IDistance;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.utility.IDataLoader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class FCM implements IObjectiveFunction {

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


	private double[][] updateMembershipValues(List<Integer> clusterCenters) {
		double[][] u = new double[this.dataLoader.getNumberOfDataPoints()][clusterCenters.size()];
		int[] centers = new int[clusterCenters.size()];
		for (int i = 0; i < clusterCenters.size(); i++) {
			centers[i] = Math.round(clusterCenters.get(i));
		}
		int clusterCount = clusterCenters.size();
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

	public float evaluate(List<Integer> candidateSolution) {
		// Evaluate the fitness of the position by FCM

		// Round the position to the nearest integer
		List<Integer> clusterCenters = candidateSolution.stream().map(Number::intValue).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);

		// Check if the fitness has already been evaluated
		if (this.fitnessCache.containsKey(clusterCenters)) {
			System.out.println("Fitness already evaluated for cluster centers: " + Arrays.toString(candidateSolution.toArray()) + " with fitness: " + fitnessCache.get(clusterCenters));
			return fitnessCache.get(clusterCenters);
		}


		double[][] u = updateMembershipValues(candidateSolution);

		float J = 0f;
		for (int i = 0; i < dataLoader.getNumberOfDataPoints(); i++) {
			double[] i_data = dataLoader.getDataPoint(i);
			for (int j = 0; j < candidateSolution.size(); j++) {
				float sum = 0;
				for (int k = 0; k < candidateSolution.size(); k++) {
					double[] k_data = dataLoader.getDataPoint(clusterCenters.get(k));
					sum += Math.pow(u[i][k], this.m) * distanceMetric.distance(k_data, i_data);
				}
				J += sum;
			}
		}


		fitnessCache.put(clusterCenters, J);

		System.out.println("Evaluated solution with cluster centers: " + Arrays.toString(candidateSolution.toArray()) + " with fitness: " + J);
		return J;


	}
}
