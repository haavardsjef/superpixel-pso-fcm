package no.haavardsjef.pso;

import no.haavardsjef.AbstractFitnessFunction;

public class Particle {
    private int numDimensions;


    private float[] position;
    private float[] velocity;
    private float fitness;
    private float[] bestPosition;
    private float bestFitness;
    private final int lowerBound;
    private final int upperBound;
    private final AbstractFitnessFunction fitnessFunction;

    public Particle(int numDimensions, int lowerBound, int upperBound, AbstractFitnessFunction fitnessFunction) {
        this.numDimensions = numDimensions;
        this.position = new float[numDimensions];
        this.velocity = new float[numDimensions];
        this.fitness = 0;
        this.bestPosition = new float[numDimensions];
        this.bestFitness = 0;
        // In our case, all dimensions have the same bounds.
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.fitnessFunction = fitnessFunction;
    }
    public float[] getPosition() {
        return position;
    }

    public float getFitness() {
        return fitness;
    }

    public void updateVelocity(float[] global_best_position, float w, float c1, float c2) {
        for (int i = 0; i < numDimensions; i++) {
            this.velocity[i] = w * this.velocity[i] + c1 * (float) Math.random() * (this.bestPosition[i] - this.position[i]) + c2 * (float) Math.random() * (global_best_position[i] - this.position[i]);
        }
    }

    public void updatePosition() {
        for (int i = 0; i < numDimensions; i++) {
            this.position[i] = this.position[i] + this.velocity[i];
            if (this.position[i] < lowerBound) {
                this.position[i] = lowerBound;
            } else if (this.position[i] > upperBound) {
                this.position[i] = upperBound;
            }
        }
    }

    public void initializeRandomly() {
        for (int i = 0; i < numDimensions; i++) {
            this.position[i] = (float) Math.random() * (upperBound - lowerBound) + lowerBound;
            this.velocity[i] = (float) Math.random();
        }
    }

    public boolean evaluate() {
        // Evaluate the fitness of the particle, returns true if the particle has improved
        this.fitness = fitnessFunction.evaluate(this.position);
        if (this.fitness > this.bestFitness) {
            this.bestFitness = this.fitness;
            this.bestPosition = this.position;
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        String s = "X: ";
        for (int i = 0; i < numDimensions; i++) {
            s += this.position[i] + " ";
        }
        return s;
    }

}
