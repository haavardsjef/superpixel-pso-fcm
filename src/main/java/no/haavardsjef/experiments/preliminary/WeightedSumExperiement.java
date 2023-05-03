package no.haavardsjef.experiments.preliminary;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.classification.ClassificationResult;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.experiments.MLFlow;
import no.haavardsjef.fcm.utility.ClusterRepresentatives;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.mlflow.api.proto.Service;
import org.mlflow.tracking.MlflowClient;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The goal of this experiment is to determine the best weights for the weighted sum criterion.
 */
@Log4j2
public class WeightedSumExperiement implements IExperiment {
    @Override
    public void runExperiment() throws IOException {
        MlflowClient client = new MlflowClient("http://35.185.118.215:8080/");

        // Set required parameters
        String experimentId = "9";

        // Define the filter string
        String filterString = "attributes.status = 'FINISHED' and params.dataset = 'indian_pines' and params.repair = 'v4'";

        // Search for active runs in the specified experiment with id "4"
        List<Service.Run> runs = client.searchRuns(List.of(experimentId), filterString, Service.ViewType.ACTIVE_ONLY, 1000).getItems();

        log.info("Found " + runs.size() + " finished runs");

        // For each run
        for (Service.Run run : runs) {
            String distanceMeasureString = "";
            String clusterCenters = "";
            String numBands = "";

            for (Service.Param param : run.getData().getParamsList()) {
                if (param.getKey().equals("clusterCentroids")) {
                    clusterCenters = param.getValue();
                }
                if (param.getKey().equals("distanceMeasure")) {
                    distanceMeasureString = param.getValue();
                }
                if (param.getKey().equals("numBands")) {
                    numBands = param.getValue();
                }
            }

            int numBandsInt = Integer.parseInt(numBands);

            // Parse clusterCenters to List<Integer>
            List<Integer> clusterCentersList = new ArrayList<>();
            // Remove brackets
            clusterCenters = clusterCenters.substring(1, clusterCenters.length() - 1);

            // Remove spaces
            clusterCenters = clusterCenters.replaceAll(" ", "");

            // Split by comma
            Arrays.stream(clusterCenters.split(",")).forEach(s -> clusterCentersList.add(Integer.parseInt(s)));
            log.info("Evaluationg clusterCenters: " + clusterCentersList);

            Dataset ds = new Dataset(DatasetName.indian_pines);
            ClusterRepresentatives cr = new ClusterRepresentatives(ds);
            cr.hardClusterBands(clusterCentersList);

            ClusterRepresentatives.RepresentativeMethod[] representativeMethods = new ClusterRepresentatives.RepresentativeMethod[]{ClusterRepresentatives.RepresentativeMethod.weightedSum, ClusterRepresentatives.RepresentativeMethod.rankingHybrid};

            MLFlow mlFlow = new MLFlow("http://35.185.118.215:8080/");

            // Initialize experiment
            mlFlow.initializeExperiment("weighted-sum-experiment");

            double w_e = 1.0;
            Double[] w_ctValues = new Double[]{0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0};
            cr.setW_e(w_e);

            for (ClusterRepresentatives.RepresentativeMethod representativeMethod : representativeMethods) {
                for (double w_ct : w_ctValues){

                    // Initialize run
                    mlFlow.startRun("w_ct-"+ w_ct + "-" + numBands);
                    mlFlow.logParam("clusterCentroids", clusterCentersList.toString());
                    mlFlow.logParam("representativeMethod", representativeMethod.toString());
                    mlFlow.logParam("dataset", ds.getDatasetName().toString());
                    mlFlow.logParam("distanceMeasure", distanceMeasureString);
                    cr.setW_ct(w_ct);
                    mlFlow.logParam("w_e", String.valueOf(w_e));
                    mlFlow.logParam("w_ct", String.valueOf(w_ct));
                    List<Integer> selectedBands = cr.selectRepresentatives(clusterCentersList, representativeMethod);
                    mlFlow.logParam("selectedBands", selectedBands.toString());
                    log.info("Selected bands: " + selectedBands);
                    mlFlow.logParam("originalRunId", run.getInfo().getRunId());
                    mlFlow.logParam("numBands", String.valueOf(numBandsInt));

                    // Evaluate using SVMClassifier
                    SVMClassifier svmClassifier = new SVMClassifier(ds);
                    int numClassificationRuns = 10;
                    mlFlow.logParam("ClassificationRuns", String.valueOf(numClassificationRuns));
                    ClassificationResult result = svmClassifier.evaluate(selectedBands, numClassificationRuns);
                    DescriptiveStatistics OA = result.getOverallAccuracy();
                    DescriptiveStatistics AOA = result.getAverageOverallAccuracy();

                    // Log metrics
                    mlFlow.logMetric("OA", OA.getMean());
                    mlFlow.logMetric("OA_SD", OA.getStandardDeviation());
                    mlFlow.logMetric("AOA", AOA.getMean());
                    mlFlow.logMetric("AOA_SD", AOA.getStandardDeviation());

                    // End run
                    mlFlow.endRun();
                }
            }
        }
    }

    public static void main(String[] args) throws IOException {
        new WeightedSumExperiement().runExperiment();
    }
}
