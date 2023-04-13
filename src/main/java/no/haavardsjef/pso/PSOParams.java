package no.haavardsjef.pso;

public class PSOParams {
	public final int numParticles;
	public final int numIterations;
	public float w;
	public float c1;
	public float c2;
	public final int numBands;

	public PSOParams(int numBands) {
		this.numParticles = 200;
		this.numIterations = 200;
		this.w = 0.5f;
		this.c1 = 0.5f;
		this.c2 = 0.2f;
		this.numBands = numBands;
	}
}
