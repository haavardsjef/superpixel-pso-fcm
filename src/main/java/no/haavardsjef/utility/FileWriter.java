package no.haavardsjef.utility;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;

public class FileWriter {


	public static boolean csvWrite(int[] data, String filename) {
		File f = new File(filename);

		try (PrintWriter pw = new PrintWriter(f)) {
			Arrays.stream(data).mapToObj(i -> String.valueOf(i)).forEach(pw::println);
			return true;
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
}
