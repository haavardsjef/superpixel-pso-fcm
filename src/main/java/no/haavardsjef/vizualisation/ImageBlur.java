package no.haavardsjef.vizualisation;

import boofcv.abst.filter.blur.BlurFilter;
import boofcv.alg.filter.blur.BlurImageOps;
import boofcv.alg.filter.blur.GBlurImageOps;
import boofcv.factory.filter.blur.FactoryBlurFilter;
import boofcv.gui.ListDisplayPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.Planar;

import java.awt.image.BufferedImage;

public class ImageBlur {
	public static void main(String[] args) {
		BufferedImage buffered = UtilImageIO.loadImageNotNull("C:\\Users\\haavahje\\github\\superpixel-pso-fcm\\band_30.png");


		Planar<GrayU8> input = ConvertBufferedImage.convertFrom(buffered, true, ImageType.pl(3, GrayU8.class));
		Planar<GrayU8> blurred = input.createSameShape();

		// Apply gaussian blur
		GBlurImageOps.gaussian(input, blurred, 0.5, -1, null);
		UtilImageIO.saveImage(blurred, "C:\\Users\\haavahje\\github\\superpixel-pso-fcm\\band_30_blurred.png");

	}
}
