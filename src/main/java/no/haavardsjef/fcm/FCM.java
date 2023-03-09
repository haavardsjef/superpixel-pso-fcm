package no.haavardsjef.fcm;

import no.haavardsjef.AbstractFitnessFunction;
import no.haavardsjef.fcm.distancemetrics.IDistance;
import no.haavardsjef.utility.IDataLoader;

import java.util.Arrays;

public class FCM extends AbstractFitnessFunction {

    private IDataLoader dataLoader;
    private float m;
    private IDistance distanceMetric;

    public FCM(float m, IDistance distanceMetric, IDataLoader dataLoader) {
        this.m = m;
        this.distanceMetric = distanceMetric;
        this.dataLoader = dataLoader;
        dataLoader.loadData();
    }



    private double[][] updateMembershipValues(float[] position){
        double[][] u = new double[this.dataLoader.getNumberOfDataPoints()][position.length];
        int[] clusterCenters = new int[position.length];
        for (int i = 0; i < position.length; i++) {
            clusterCenters[i] = Math.round(position[i]);
        }

        int clusterCount = position.length;
        for (int i = 0; i < dataLoader.getNumberOfDataPoints(); i++) {
            double[] i_data = dataLoader.getDataPoint(i);
            for (int j = 0; j < clusterCount; j++) {
                if (i == clusterCenters[j]){
                    u[i][j] = 1;
                    continue;
                }


                float sum = 0;
                double[] j_data = dataLoader.getDataPoint(clusterCenters[j]);
                double upper = distanceMetric.distance(i_data, j_data);
                for (int k = 0; k < clusterCount; k++) {
                    double[] k_data = dataLoader.getDataPoint(clusterCenters[k]);
                    double lower = distanceMetric.distance(i_data, k_data);
                    sum += Math.pow((upper/lower), 2/(this.m -1));
                }
                u[i][j] = 1/sum; // TODO: Cache this
            }
        }
        return u;
    }
    public float evaluate(float[] position) {
        // Evaluate the fitness of the position by FCM

        // Round the position to the nearest integer
        int[] clusterCenters = new int[position.length];
        for (int i = 0; i < position.length; i++) {
            clusterCenters[i] = Math.round(position[i]);
        }

        double[][] u = updateMembershipValues(position);

        float J = 0f;
        for (int i = 0; i < dataLoader.getNumberOfDataPoints(); i++) {
            double[] i_data = dataLoader.getDataPoint(i);
            for (int j = 0; j < position.length; j++) {
                float sum = 0;
                for (int k = 0; k < position.length; k++) {
                    double[] k_data = dataLoader.getDataPoint(clusterCenters[k]);
                    sum += Math.pow(u[i][k], this.m) * distanceMetric.distance(k_data, i_data);
                }
                J += sum;
            }
        }
        System.out.println("Evaluated solution with cluster centers: " + Arrays.toString(clusterCenters) + " with fitness: " + J);
        return J;


    }
}
