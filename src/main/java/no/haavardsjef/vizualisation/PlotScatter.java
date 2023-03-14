package no.haavardsjef.vizualisation;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PlotScatter {

	public void plot(double[][] data, List<Integer> highlight) {
		// Typically highlight contains the indices of the cluster centers.
		// Create x and y lists
		List<Double> x = new ArrayList<>();
		List<Double> x2 = new ArrayList<>();
		List<Double> y = new ArrayList<>();
		List<Double> y2 = new ArrayList<>();

		for (int i = 0; i < data.length; i++) {
			if (highlight.contains(i)) {
				x2.add(data[i][0]);
				y2.add(data[i][1]);
			} else {
				x.add(data[i][0]);
				y.add(data[i][1]);
			}
		}

		Plot plt = Plot.create();
		plt.plot().add(x, y, "o");
		plt.plot().add(x2, y2, "x");

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}
}
