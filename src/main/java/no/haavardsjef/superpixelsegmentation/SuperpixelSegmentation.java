package no.haavardsjef.superpixelsegmentation;

import boofcv.abst.segmentation.ImageSuperpixels;
import boofcv.alg.filter.blur.GBlurImageOps;
import boofcv.alg.segmentation.ComputeRegionMeanColor;
import boofcv.alg.segmentation.ImageSegmentationOps;
import boofcv.factory.segmentation.ConfigFh04;
import boofcv.factory.segmentation.ConfigSlic;
import boofcv.factory.segmentation.FactoryImageSegmentation;
import boofcv.factory.segmentation.FactorySegmentationAlg;
import boofcv.gui.ListDisplayPanel;
import boofcv.gui.feature.VisualizeRegions;
import boofcv.gui.image.ShowImages;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.feature.ColorQueue_F32;
import boofcv.struct.image.*;
import org.ddogleg.struct.DogArray_I32;

import java.awt.image.BufferedImage;

public class SuperpixelSegmentation {
	/**
	 * Segments the image
	 */
	public static <T extends ImageBase<T>> int[] performSegmentation(ImageSuperpixels<T> alg, T color) {
		// Segmentation often works better after blurring the image. Reduces high frequency image components which
		// can cause over segmentation
		GBlurImageOps.gaussian(color, color, 0.5, -1, null); //TODO: Explore options here

		// Storage for segmented image. Each pixel will be assigned a label from 0 to N-1, where N is the number
		// of segments in the image
		var pixelToSegment = new GrayS32(color.width, color.height);

		// Segmentation magic happens here
		alg.segment(color, pixelToSegment);

		// Displays the results
		visualize(pixelToSegment, color, alg.getTotalSuperpixels());

		int[] flatSuperpixelMap = pixelToSegment.data;
		int numberOfSuperpixels = alg.getTotalSuperpixels();
		System.out.println("Segmented image into " + numberOfSuperpixels + " superpixels");
		return flatSuperpixelMap;

	}

	/**
	 * Visualizes results three ways. 1) Colorized segmented image where each region is given a random color.
	 * 2) Each pixel is assigned the mean color through out the region. 3) Black pixels represent the border
	 * between regions.
	 */
	public static <T extends ImageBase<T>>
	void visualize(GrayS32 pixelToRegion, T color, int numSegments) {
		// Computes the mean color inside each region
		ImageType<T> type = color.getImageType();
		ComputeRegionMeanColor<T> colorize = FactorySegmentationAlg.regionMeanColor(type);

		var segmentColor = new ColorQueue_F32(type.getNumBands());
		segmentColor.resize(numSegments);

		var regionMemberCount = new DogArray_I32();
		regionMemberCount.resize(numSegments);

		ImageSegmentationOps.countRegionPixels(pixelToRegion, numSegments, regionMemberCount.data);
		colorize.process(color, pixelToRegion, regionMemberCount, segmentColor);

		// Draw each region using their average color
		BufferedImage outColor = VisualizeRegions.regionsColor(pixelToRegion, segmentColor, null);
		// Draw each region by assigning it a random color
		BufferedImage outSegments = VisualizeRegions.regions(pixelToRegion, numSegments, null);

		// Make region edges appear red
		var outBorder = new BufferedImage(color.width, color.height, BufferedImage.TYPE_INT_RGB);
		ConvertBufferedImage.convertTo(color, outBorder, true);
		VisualizeRegions.regionBorders(pixelToRegion, 0xFF0000, outBorder);

		// Show the visualization results
		var gui = new ListDisplayPanel();
		gui.addImage(outColor, "Color of Segments");
		gui.addImage(outBorder, "Region Borders");
		gui.addImage(outSegments, "Regions");
		ShowImages.showWindow(gui, "Superpixels", true);
	}

	public int[] segment(Planar<GrayF32> image) {
		ImageType<Planar<GrayF32>> imageType = ImageType.pl(3, GrayF32.class);
		ImageSuperpixels algorithm = FactoryImageSegmentation.slic(new ConfigSlic(100, 200f), imageType);
		return performSegmentation(algorithm, image);
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