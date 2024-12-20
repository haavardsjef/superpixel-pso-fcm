package no.haavardsjef.pso;

public class PSOParams {
	public final int numParticles;
	public final int numIterations;
	public float w;
	public float c1;
	public float c2;
	public final int numBands;

	public PSOParams(int numBands) {
		this.numParticles = 100;
		this.numIterations = 100;
		this.w = 0.7f;
		this.c1 = 1.0f;
		this.c2 = 1.0f;
		this.numBands = numBands;
	}
}
