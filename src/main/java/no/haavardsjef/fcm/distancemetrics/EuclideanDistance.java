package no.haavardsjef.fcm.distancemetrics;

public class EuclideanDistance implements IDistance {

    @Override
    public double distance(double[] data1, double[] data2) {
        double result = 0;
        for (int i = 0; i < data1.length; i++) {
            result += Math.pow(data1[i] - data2[i], 2);
        }
        return Math.sqrt(result);
    }
}
