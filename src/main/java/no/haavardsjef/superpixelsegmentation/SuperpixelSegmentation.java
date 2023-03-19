package no.haavardsjef.superpixelsegmentation;

import boofcv.abst.segmentation.ImageSuperpixels;
import boofcv.alg.filter.blur.GBlurImageOps;
import boofcv.factory.segmentation.ConfigFh04;
import boofcv.factory.segmentation.FactoryImageSegmentation;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.*;

import java.awt.image.BufferedImage;

public class SuperpixelSegmentation {
	/**
	 * Segments the image
	 */
	public static <T extends ImageBase<T>> int[] performSegmentation(ImageSuperpixels<T> alg, T color) {
		// Segmentation often works better after blurring the image. Reduces high frequency image components which
		// can cause over segmentation
		GBlurImageOps.gaussian(color, color, 0.5, -1, null);

		// Storage for segmented image. Each pixel will be assigned a label from 0 to N-1, where N is the number
		// of segments in the image
		var pixelToSegment = new GrayS32(color.width, color.height);

		// Segmentation magic happens here
		alg.segment(color, pixelToSegment);

		int[] flatSuperpixelMap = pixelToSegment.data;
		int numberOfSuperpixels = alg.getTotalSuperpixels();

		return flatSuperpixelMap;

	}

	public void segment(Planar<GrayF32> image) {
		return;
	}


	public static void main(String[] args) {
		BufferedImage image = UtilImageIO.loadImageNotNull(UtilIO.pathExample("C:\\Users\\haavahje\\github\\superpixel-pso-fcm\\segmentation\\raw.jpg"));

		// you probably don't want to segment along the image's alpha channel and the code below assumes 3 channels
		image = ConvertBufferedImage.stripAlphaChannel(image);

		// Select input image type. Some algorithms behave different depending on image type
		ImageType<Planar<GrayF32>> imageType = ImageType.pl(3, GrayF32.class);
//		ImageType<Planar<GrayU8>> imageType = ImageType.pl(3, GrayU8.class);
//		ImageType<GrayF32> imageType = ImageType.single(GrayF32.class);
//		ImageType<GrayU8> imageType = ImageType.single(GrayU8.class);

//		ImageSuperpixels alg = FactoryImageSegmentation.meanShift(null, imageType);
//		ImageSuperpixels alg = FactoryImageSegmentation.slic(new ConfigSlic(400), imageType);
		ImageSuperpixels alg = FactoryImageSegmentation.fh04(new ConfigFh04(100, 30), imageType);
//		ImageSuperpixels alg = FactoryImageSegmentation.watershed(null, imageType);

		// Convert image into BoofCV format
		ImageBase color = imageType.createImage(image.getWidth(), image.getHeight());
		ConvertBufferedImage.convertFrom(image, color, true);

		// Segment the image
		performSegmentation(alg, color);
	}
}