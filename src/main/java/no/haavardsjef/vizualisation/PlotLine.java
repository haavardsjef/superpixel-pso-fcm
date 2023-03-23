package no.haavardsjef.vizualisation;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public abstract class PlotLine {


	public static void plot(List<Double> data) {
		// Create x and y lists
		List<Double> x = new ArrayList<>();

		for (int i = 0; i < data.size(); i++) {
			x.add((double) i);
		}


		// Plot swarm using matplotlib4j
		Plot plt = Plot.create();
		plt.plot().add(x, data);

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}


}
