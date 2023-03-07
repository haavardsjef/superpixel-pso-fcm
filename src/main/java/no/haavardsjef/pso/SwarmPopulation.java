package no.haavardsjef.pso;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SwarmPopulation {

    public ArrayList<Particle> particles;
    public int numParticles;
    public int numDimensions;
    public float[] globalBestPosition;
    public float globalBestFitness;
    public int lowerBound;
    public int upperBound;

    public SwarmPopulation(int numParticles, int numDimensions, int lowerBound, int upperBound) {
        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.particles = new ArrayList<>(numParticles);
        this.globalBestPosition = new float[numDimensions];
        this.globalBestFitness = 0;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;

        for (int i = 0; i < numParticles; i++) {
            Particle particle = new Particle(numDimensions, lowerBound, upperBound, new Squared());
            particle.initializeRandomly();
            particles.add(particle);
        }
    }

    public void optimize(int numIterations, float w, float c1, float c2) {
        for (int i = 0; i < numIterations; i++) {
            System.out.println("Iteration: " + i);
            for (Particle particle : particles) {
                particle.updateVelocity(globalBestPosition, w, c1, c2);
                particle.updatePosition();
                if (particle.evaluate()) {
                    if (particle.getFitness() > globalBestFitness) {
                        globalBestFitness = particle.getFitness();
                        globalBestPosition = particle.getPosition();
                    }
                }
            }
        }
        System.out.println("Global best fitness: " + globalBestFitness);
        System.out.println("Global best position: " + globalBestPosition);
    }

    public static void main(String[] args) {
        SwarmPopulation swarmPopulation = new SwarmPopulation(50, 10, 0, 10);
        // Start timer
        long startTime = System.nanoTime();
        swarmPopulation.optimize(100, 2f, 0.5f, 0.5f);
        // Stop timer
        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1000000;
        System.out.println("Time: " + duration + "ms");


    }
}
