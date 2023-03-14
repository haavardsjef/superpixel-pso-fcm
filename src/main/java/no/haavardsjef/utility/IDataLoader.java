package no.haavardsjef.utility;

public interface IDataLoader {
    public void loadData();

    /** Returns a singular datapoint from the loaded dataset.
     * loadData() needs to be called before this.
     * @param index - the index of the datapoint
     * @return array of doubles containing the data values for every dimension of that singular datapoint.
     */
    public double[] getDataPoint(int index);


    /** Returns the number of data points available in the loaded dataset.
     * loadData() needs to be called before this.
     * @return The number of data points available in the loaded dataset.
     */
    public int getNumberOfDataPoints();
}
