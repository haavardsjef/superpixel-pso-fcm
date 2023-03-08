package no.haavardsjef.pso;

import no.haavardsjef.AbstractFitnessFunction;

public class Squared extends AbstractFitnessFunction {
    @Override
    public float evaluate(float[] position) {
        float fitness = 0;
        for (int i = 0; i < position.length; i++) {
            fitness += position[i] * position[i];
        }
        return fitness;
    }
}