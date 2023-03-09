package no.haavardsjef.utility;

public interface IDataLoader {
    public void loadData();
    public double[] getDataPoint(int index);

    public int getNumberOfDataPoints();
}
