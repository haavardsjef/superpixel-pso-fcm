package no.haavardsjef.classification;

public class McNemarsTest {

	public static double computeTestStatistic(int[][] matrix) {
		// The test statistic is computed as (b-c)^2 / (b+c),
		// where b and c are the off-diagonal elements of the matrix
		double b = matrix[0][1];
		double c = matrix[1][0];

//		return Math.pow((b - c), 2) / (b + c);
		return (b - c) / Math.sqrt(b + c);
	}

	public static double computePValue(int[][] matrix) {

		double testStatistic = computeTestStatistic(matrix);

		// Using standard normal distribution to get p-value
		org.apache.commons.math3.distribution.NormalDistribution normal =
				new org.apache.commons.math3.distribution.NormalDistribution();

		double pValue = 2 * (1 - normal.cumulativeProbability(Math.abs(testStatistic)));
		return pValue;
	}

	public static void main(String[] args) {
		int[][] matrix = {{100, 50}, {30, 120}};
		McNemarsTest mcnemarsTest = new McNemarsTest();
		double testStatistic = computeTestStatistic(matrix);
		double pValue = computePValue(matrix);

		System.out.println("Test statistic: " + testStatistic);
		System.out.println("P-value: " + pValue);
	}
}
