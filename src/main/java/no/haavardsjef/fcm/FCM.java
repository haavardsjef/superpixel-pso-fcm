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
        return 0f;
    }

    private float membership(int clusterCenter, int band){
        // TODO: Check if band is a cluster center
        float result = 0f;

        float numerator = 0f;
        // Iterate every band
        // TODO: THis might be iterate every cluster center instead of every band
        for (int i = 0; i < this.dataLoader.getNumBands(); i++) {
            numerator += Math.pow(this.distance(i, band) , 2f / (this.m - 1f));
        }

        double denominator = Math.pow(this.distance(clusterCenter, band) , 2f / (this.m - 1f));

        result = (float) (numerator / denominator);

        if (result < 0f){
            throw new RuntimeException("membership is negative");
        } else if (result > 1f){
            throw new RuntimeException("membership is over 1");
        }
        return result;
    }

    private float clusterCenterCostFunction(int clusterCenter){
        // Calculate individual contribution of a single cluster center.
        float J = 0f;

        // Iterate every band
        for (int i = 0; i < this.dataLoader.getNumBands(); i++) {
            J += Math.pow(this.membership(clusterCenter, i), this.m) * this.distance(clusterCenter, i); // TODO
        }



        return 0f;

    }

    private float evaluate_cluster_centers(int[] clusterCenters) {
        float evaluation = 0f;
        for (int clusterCenter : clusterCenters) {
            evaluation += this.clusterCenterCostFunction(clusterCenter);
        }
        return evaluation;
    }
    public float evaluate(float[] position) {
        // Evaluate the fitness of the position by FCM

        // Round the position to the nearest integer
        int[] clusterCenters = new int[position.length];
        for (int i = 0; i < position.length; i++) {
            clusterCenters[i] = Math.round(position[i]);
        }

        return this.evaluate_cluster_centers(clusterCenters);

    }
}
