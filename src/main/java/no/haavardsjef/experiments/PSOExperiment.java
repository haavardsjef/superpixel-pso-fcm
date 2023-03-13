package no.haavardsjef.experiments;

import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.objectivefunctions.SquaredObjectiveFunction;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DataLoader;

public class PSOExperiment implements IExperiment {

    public void runExperiment() {
        IObjectiveFunction objectiveFunction = new SquaredObjectiveFunction();
        Bounds bounds = new Bounds(-100, 100);
        SwarmPopulation swarmPopulation = new SwarmPopulation(100, 2, bounds, objectiveFunction);
        float[] solution = swarmPopulation.optimize(50, 0.5f, 0.5f, 0.2f, true);
    }


    public static void main(String[] args) {
        PSOExperiment psoExperiment = new PSOExperiment();
        psoExperiment.runExperiment();
    }
}
