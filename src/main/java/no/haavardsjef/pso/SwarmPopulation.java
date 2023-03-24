package no.haavardsjef.pso;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.vizualisation.PlotLine;
import no.haavardsjef.vizualisation.Visualizations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

@Log4j2
public class SwarmPopulation {

	public ArrayList<Particle> particles;
	public int numParticles;
	public int numDimensions;
	public float[] globalBestPosition;
	public float globalBestFitness;
	private Bounds bounds;

	public IObjectiveFunction objectiveFunction;
	private Particle solution;

	public SwarmPopulation(int numParticles, int numDimensions, Bounds bounds, IObjectiveFunction objectiveFunction) {
		this.numParticles = numParticles;
		this.numDimensions = numDimensions;
		this.particles = new ArrayList<>(numParticles);
		this.globalBestPosition = new float[numDimensions];
		this.globalBestFitness = Float.POSITIVE_INFINITY;
		this.bounds = bounds;
		this.objectiveFunction = objectiveFunction;


		for (int i = 0; i < numParticles; i++) {
			Particle particle = new Particle(numDimensions, bounds, this.objectiveFunction);
			particle.initializeRandomly();
			particles.add(particle);
		}
	}

	public Particle optimize(int numIterations, float w, float c1, float c2, boolean plot) {
		AtomicInteger iterationsSinceImprovement = new AtomicInteger();
		long startTime = System.nanoTime();
		List<Double> avgFitness = new ArrayList<Double>();
		for (int i = 0; i < numIterations; i++) {
			iterationsSinceImprovement.getAndIncrement();

			if (iterationsSinceImprovement.get() > 10) {
				log.info("No improvement in 10 iterations. Stopping optimization.");
				break;
			}


			System.out.println("Iteration: " + i);
			if (plot && this.numDimensions == 2) {
				Visualizations.plotSwarm(this.particles, i, this.bounds);
			}
			float totalFitness = particles.stream().map(particle -> {
				particle.updateVelocity(globalBestPosition, w, c1, c2);
				particle.updatePosition();

				float p_fitness = particle.evaluate();

				if (p_fitness < globalBestFitness) {
					iterationsSinceImprovement.set(0);
					globalBestFitness = particle.getFitness();
					globalBestPosition = particle.getPosition().clone();
					this.solution = particle;
				}
				return particle.getFitness();
			}).reduce(0f, Float::sum);
			avgFitness.add((double) (totalFitness / this.numParticles));
			System.out.println("Global best fitness after iteration " + i + ": " + globalBestFitness);
			System.out.println("Global best position after iteration " + i + ": " + Arrays.toString(globalBestPosition));
		}
		if (plot && this.numDimensions == 2) {
			Visualizations.plotSwarm(this.particles, numIterations, this.bounds);
		}
		long endTime = System.nanoTime();
		long duration = (endTime - startTime) / 1000000000;
		System.out.println("Elapsed time for optimization: " + duration + "s");

		if (plot) {
			PlotLine.plot(avgFitness);
		}


		System.out.println("Global best fitness: " + globalBestFitness);
		System.out.println("Global best position: " + Arrays.toString(globalBestPosition));
		return solution;
	}
}
