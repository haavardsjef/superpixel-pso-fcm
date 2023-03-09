package no.haavardsjef;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.DataLoader;

public class App {


    public static void main(String[] args) {
        AbstractFitnessFunction fitnessFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
        SwarmPopulation swarmPopulation = new SwarmPopulation(30, 2, 0, 199, fitnessFunction);
        // Start timer
        long startTime = System.nanoTime();
        swarmPopulation.optimize(30, 0.5f, 0.5f, 0.2f);
        // Stop timer
        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1000000;
        System.out.println("Time: " + duration + "ms");
    }
}
