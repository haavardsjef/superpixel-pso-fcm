package no.haavardsjef.pso;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import no.haavardsjef.AbstractFitnessFunction;
import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.utility.DataLoader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SwarmPopulation {

    public ArrayList<Particle> particles;
    public int numParticles;
    public int numDimensions;
    public float[] globalBestPosition;
    public float globalBestFitness;
    public int lowerBound;
    public int upperBound;

    public AbstractFitnessFunction fitnessFunction;

    public SwarmPopulation(int numParticles, int numDimensions, int lowerBound, int upperBound, AbstractFitnessFunction fitnessFunction) {
        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.particles = new ArrayList<>(numParticles);
        this.globalBestPosition = new float[numDimensions];
        this.globalBestFitness = Float.POSITIVE_INFINITY;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.fitnessFunction = fitnessFunction;



        for (int i = 0; i < numParticles; i++) {
            Particle particle = new Particle(numDimensions, lowerBound, upperBound, this.fitnessFunction);
            particle.initializeRandomly();
            particles.add(particle);
        }
    }

    private void plot(int iteration){
        List<Double> x = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        for (Particle particle : this.particles) {
            x.add((double) particle.getPosition()[0]);
            y.add((double) particle.getPosition()[1]);
        }

        // Plot swarm using matplotlib4j
        Plot plt = Plot.create();
        plt.plot().add(x, y, "o");
        // Make both x and y go from 0 to 10
        plt.xlim(0, 200);
        plt.ylim(0, 200);
        plt.title("Iteration " + iteration);
//            plt.show();
        plt.savefig("C:/Users/Haavard/github/superpixel-pso-fcm/viz/" + iteration + ".png");
        try {
            plt.executeSilently();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Saved plot " + iteration);

    }

    public void optimize(int numIterations, float w, float c1, float c2, boolean plot) {
        for (int i = 0; i < numIterations; i++) {
            System.out.println("Iteration: " + i);
            if(plot && this.numDimensions == 2){
                plot(i);
            }
            for (Particle particle : particles) {
                particle.updateVelocity(globalBestPosition, w, c1, c2);
                particle.updatePosition();
                if (particle.evaluate()) {
                    if (particle.getFitness() < globalBestFitness) {
                        globalBestFitness = particle.getFitness();
                        globalBestPosition = particle.getPosition();
                    }
                }
            }
            System.out.println("Global best fitness after iteration " + i + ": " + globalBestFitness);
            System.out.println("Global best position after iteration " + i + ": " + Arrays.toString(globalBestPosition));
        }
        if(plot && this.numDimensions == 2){
            plot(numIterations);
        }
        System.out.println("Global best fitness: " + globalBestFitness);
        System.out.println("Global best position: " + Arrays.toString(globalBestPosition));
    }
}
