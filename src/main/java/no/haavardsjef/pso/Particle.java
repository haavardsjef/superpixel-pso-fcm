package no.haavardsjef.pso;

import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.utility.Bounds;

import java.util.ArrayList;
import java.util.List;

public class Particle {
	private int numDimensions;


	private final float[] position;
	private float[] velocity;
	private float fitness;
	private float[] bestPosition;
	private float bestFitness;
	private final Bounds bounds;
	private final IObjectiveFunction objectiveFunction;

	public Particle(int numDimensions, Bounds bounds, IObjectiveFunction objectiveFunction) {
		this.numDimensions = numDimensions;
		this.position = new float[numDimensions];
		this.velocity = new float[numDimensions];
		// Fitness starts as infinity, so that the first evaluation will always improve it.
		this.fitness = Float.POSITIVE_INFINITY;
		this.bestPosition = new float[numDimensions];
		this.bestFitness = Float.POSITIVE_INFINITY;
		// In our case, all dimensions have the same bounds.
		this.bounds = bounds;
		this.objectiveFunction = objectiveFunction;
	}

	public Particle(float[] position, Bounds bounds, IObjectiveFunction objectiveFunction) {
		this.position = position;
		this.bounds = bounds;
		this.objectiveFunction = objectiveFunction;
	}

	public float[] getPosition() {
		return position;
	}

	public float getFitness() {
		return fitness;
	}

	public List<Integer> getDiscretePositionSorted() {
		List<Integer> discretePosition = new ArrayList<>(this.numDimensions);
		for (float f : this.position) {
			discretePosition.add(Math.round(f));
		}
		discretePosition.sort(Integer::compareTo);
		return discretePosition;

	}

	public void updateVelocity(float[] global_best_position, float w, float c1, float c2) {
		for (int i = 0; i < numDimensions; i++) {
			this.velocity[i] = w * this.velocity[i] + c1 * (float) Math.random() * (this.bestPosition[i] - this.position[i]) + c2 * (float) Math.random() * (global_best_position[i] - this.position[i]);
		}
	}

	public void updatePosition() {
		for (int i = 0; i < numDimensions; i++) {
			this.position[i] = this.position[i] + this.velocity[i];
			if (this.position[i] < bounds.lower()) {
				this.position[i] = bounds.lower();
			} else if (this.position[i] > bounds.upper()) {
				this.position[i] = bounds.upper();
			}
		}
	}

	public void initializeRandomly() {
		for (int i = 0; i < numDimensions; i++) {
			this.position[i] = (float) Math.random() * (bounds.upper() - bounds.lower()) + bounds.lower();
			this.velocity[i] = (float) Math.random();
		}
	}

	public float evaluate() {
		// Evaluate the fitness of the particle, returns true if the particle has improved
		float newFitness = objectiveFunction.evaluate(this.getDiscretePositionSorted());
		if (newFitness < this.bestFitness) {
			this.bestFitness = newFitness;
			this.bestPosition = this.position;
		}
		this.fitness = newFitness;
		return newFitness;
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
