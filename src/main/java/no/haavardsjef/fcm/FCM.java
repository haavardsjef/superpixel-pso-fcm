package no.haavardsjef.fcm;

import no.haavardsjef.AbstractFitnessFunction;
import no.haavardsjef.utility.DataLoader;

import java.util.Arrays;

public class FCM extends AbstractFitnessFunction {

    private DataLoader dataLoader;
    private float m;

    public FCM(float m) {
        this.dataLoader = new DataLoader();
        dataLoader.loadData();
        this.m = m;
    }


    private float distance(int band1, int band2){
        double[] band1_data = this.dataLoader.getSpecificBandFlatted(band1);
        double[] band2_data = this.dataLoader.getSpecificBandFlatted(band2);

        float result = 0f;

        for (int i = 0; i < band1_data.length; i++) {
            result += Math.abs(band1_data[i] - band2_data[i]); // TODO: Check if this is correct
        }
        return result;
    }


    private double[][] updateMembershipValues(float[] position){
        double[][] u = new double[this.dataLoader.getNumBands()][position.length];
        int[] clusterCenters = new int[position.length];
        for (int i = 0; i < position.length; i++) {
            clusterCenters[i] = Math.round(position[i]);
        }

        int clusterCount = position.length;
        for (int i = 0; i < dataLoader.getNumBands(); i++) {
            for (int j = 0; j < clusterCount; j++) {
                if (i == clusterCenters[j]){
                    u[i][j] = 1;
                    continue;
                }


                float sum = 0;
                float upper = this.distance(i, clusterCenters[j]);
                for (int k = 0; k < clusterCount; k++) {
                    float lower = this.distance(i, clusterCenters[k]);
                    sum += Math.pow((upper/lower), 2/(this.m -1));
                }
                u[i][j] = 1/sum;
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
        for (int i = 0; i < dataLoader.getNumBands(); i++) {
            for (int j = 0; j < position.length; j++) {
                float sum = 0;
                for (int k = 0; k < position.length; k++) {
                    sum += Math.pow(u[i][k], this.m) * this.distance(clusterCenters[k], i);
                }
                J += sum;
            }
        }
        System.out.println("Evaluated solution with cluster centers: " + Arrays.toString(clusterCenters) + " with fitness: " + J);
        return J;


    }
}
