package no.haavardsjef.experiments;

import no.haavardsjef.pso.PSOParams;
import org.mlflow.api.proto.Service;
import org.mlflow.tracking.MlflowClient;

import java.util.Optional;

public class MLFlow {

	private final String trackingUri = "http://localhost:5000";
	private final MlflowClient mlflowClient;
	private Optional<String> experimentId;
	private boolean activeRun = false;
	private String runId;

	public MLFlow() {
		this.mlflowClient = new MlflowClient(trackingUri);
	}

	public MLFlow(String trackingUri) {
		this.mlflowClient = new MlflowClient(trackingUri);
	}

	/**
	 * Creates new experiment if it does not exist,
	 * otherwise loads the existing experiment
	 *
	 * @param experimentName The name of the experiment to initialize
	 */
	public void initializeExperiment(String experimentName) {
		Optional<String> experimentId = this.mlflowClient.getExperimentByName(experimentName).map(Service.Experiment::getExperimentId);
		if (!experimentId.isPresent()) {
			experimentId = Optional.of(this.mlflowClient.createExperiment(experimentName));
		}
		this.experimentId = experimentId;
	}

	/**
	 * Starts a new run
	 *
	 * @param runName The name of the run to start
	 */
	public void startRun(String runName) {
		Service.RunInfo runInfo = this.mlflowClient.createRun(this.experimentId.get());
		String runId = runInfo.getRunId();

		// Set the run name
		this.mlflowClient.setTag(runId, "mlflow.runName", runName);
		this.runId = runId;
		this.activeRun = true;
	}

	/**
	 * Logs a parameter to the active run
	 *
	 * @param key   The name of the parameter
	 * @param value The value of the parameter
	 */
	public void logParam(String key, String value) {
		if (this.activeRun) {
			this.mlflowClient.logParam(this.runId, key, value);
		} else {
			throw new IllegalStateException("No active run");
		}
	}

	/**
	 * Logs a metric to the active run
	 *
	 * @param key   The name of the metric
	 * @param value The value of the metric
	 */
	public void logMetric(String key, double value) {
		if (this.activeRun) {
			this.mlflowClient.logMetric(this.runId, key, value);
		} else {
			throw new IllegalStateException("No active run");
		}
	}

	/**
	 * Ends the active run
	 */
	public void endRun() {
		this.mlflowClient.setTerminated(runId, Service.RunStatus.FINISHED);
		this.activeRun = false;
		this.runId = null;
	}

	public void logPSOParams(PSOParams params) {
		this.logParam("numParticles", String.valueOf(params.numParticles));
		this.logParam("numIterations", String.valueOf(params.numIterations));
		this.logParam("w", String.valueOf(params.w));
		this.logParam("c1", String.valueOf(params.c1));
		this.logParam("c2", String.valueOf(params.c2));
		this.logParam("numBands", String.valueOf(params.numBands));

	}


}
