package no.haavardsjef.classification;

public class McNemarsTest {

	public double computeTestStatistic(int[][] matrix) {
		// The test statistic is computed as (b-c)^2 / (b+c),
		// where b and c are the off-diagonal elements of the matrix
		double b = matrix[0][1];
		double c = matrix[1][0];

		return Math.pow((b - c), 2) / (b + c);
	}

	public double computePValue(int[][] matrix) {
		// The p-value is computed using the chi-square distribution with 1 degree of freedom
		org.apache.commons.math3.distribution.ChiSquaredDistribution chiSquare =
				new org.apache.commons.math3.distribution.ChiSquaredDistribution(1);

		double testStatistic = computeTestStatistic(matrix);

		// The method `cumulativeProbability` gives the probability that a random variable
		// with this distribution takes a value less than or equal to `testStatistic`,
		// so we subtract it from 1 to get the probability of a greater value.
		return 1 - chiSquare.cumulativeProbability(testStatistic);
	}

	public static void main(String[] args) {
		int[][] matrix = {{100, 50}, {30, 120}};
		McNemarsTest mcnemarsTest = new McNemarsTest();
		double testStatistic = mcnemarsTest.computeTestStatistic(matrix);
		double pValue = mcnemarsTest.computePValue(matrix);

		System.out.println("Test statistic: " + testStatistic);
		System.out.println("P-value: " + pValue);
	}
}
