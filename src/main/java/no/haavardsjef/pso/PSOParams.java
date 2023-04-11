package no.haavardsjef.pso;

public class PSOParams {
	public final int numParticles;
	public final int numIterations;
	public final float w;
	public final float c1;
	public final float c2;
	public final int numBands;

	public PSOParams(int numBands) {
		this.numParticles = 100;
		this.numIterations = 100;
		this.w = 0.5f;
		this.c1 = 0.5f;
		this.c2 = 0.2f;
		this.numBands = numBands;
	}
}
